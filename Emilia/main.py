# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import librosa
import numpy as np
import sys
import os
import tqdm
import warnings
import torch
import logging
from pydub import AudioSegment
from pyannote.audio import Pipeline, Annotation
import pandas as pd

from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad

warnings.filterwarnings("ignore")
audio_count = 0


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)

    logger.debug("Audio file converted to WAV format")

    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }


@time_logger
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals.
    """
    rate = audio["sample_rate"]
    mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]

    return audio


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization on the given audio.
    FIX: Now processes audio in chunks to prevent CUDA Out of Memory errors.
    """
    logger.debug("Start speaker diarization")
    waveform_full = audio['waveform']
    sample_rate = audio['sample_rate']
    
    chunk_duration_s = 30.0  # Process in 30-second chunks
    chunk_samples = int(chunk_duration_s * sample_rate)
    
    full_annotation = Annotation()
    
    print("  Diarizing in chunks to conserve memory...")
    for start_sample in tqdm.tqdm(range(0, len(waveform_full), chunk_samples), desc="Diarization Chunks"):
        end_sample = start_sample + chunk_samples
        chunk_waveform = waveform_full[start_sample:end_sample]
        
        # Skip empty chunks
        if len(chunk_waveform) == 0:
            continue

        # Prepare chunk for the pipeline
        chunk_for_pipeline = {
            "waveform": torch.tensor(chunk_waveform).unsqueeze(0).to(device),
            "sample_rate": sample_rate,
        }
        
        # Get diarization for the current chunk
        chunk_annotation = dia_pipeline(chunk_for_pipeline)

        # Shift the segment times in the chunk's annotation to be relative to the full audio
        for segment, track, label in chunk_annotation.itertracks(yield_label=True):
            start_time_global = start_sample / sample_rate + segment.start
            end_time_global = start_sample / sample_rate + segment.end
            full_annotation[segment.shift(start_sample / sample_rate)] = label

    # Merge overlapping segments from different chunks
    full_annotation = full_annotation.support()

    diarize_df = pd.DataFrame(
        full_annotation.itertracks(yield_label=True),
        columns=["segment", "track", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels.
    """
    MERGE_GAP = 2
    MIN_SEGMENT_LENGTH = 3
    MAX_SEGMENT_LENGTH = 30

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def asr(vad_segments, audio):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments.
    """
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]

    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        unique_languages = list(set(valid_vad_segments_language))
        for language_token in unique_languages:
            language = language_token
            vad_segments_lang = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments_lang,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            result = transcribe_result_temp["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []


@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio segments.
    """
    audio_waveform = audio["waveform"]
    sample_rate = 16000

    audio_resampled = librosa.resample(
        audio_waveform, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio_resampled[start:end]
        dnsmos_score = dnsmos_compute_score(segment, sample_rate, False)["SIG"]
        vad_list[index]["dnsmos"] = dnsmos_score

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])
    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")
    return predict_dnsmos, vad_list


def filter_segments(mos_list):
    """
    Filter out segments based on quality metrics.
    """
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(mos_list)
    filtered_segment_count = len(filtered_audio_stats)
    all_segment_count = len(all_audio_stats)
    
    if all_segment_count > 0:
        logger.debug(
            f"> {all_segment_count - filtered_segment_count}/{all_segment_count} "
            f"({(all_segment_count - filtered_segment_count) / all_segment_count:.2%}) segments filtered."
        )
    
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]
    all_list = [mos_list[idx] for idx, _ in all_audio_stats]
    return filtered_list, all_list


def main_process(audio_path, save_path=None, audio_name=None):
    """
    Main processing function for a single audio file.
    """
    if not any(audio_path.endswith(ext) for ext in (".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
        return None, None

    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    
    if save_path is None:
        processed_folder = os.path.dirname(audio_path) + "_processed"
        save_path = os.path.join(processed_folder, audio_name)
    
    os.makedirs(save_path, exist_ok=True)
    logger.debug(f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}")

    logger.info("Step 0: Preprocessing")
    audio = standardization(audio_path)

    logger.info("Step 1: Source Separation")
    audio = source_separation(separate_predictor1, audio)

    logger.info("Step 2: Speaker Diarization")
    speakerdia = speaker_diarization(audio)

    logger.info("Step 3: VAD Segmentation")
    vad_list = vad.vad(speakerdia, audio)
    segment_list = cut_by_speaker_label(vad_list)

    logger.info("Step 4: ASR")
    asr_result = asr(segment_list, audio)
    if not asr_result:
        logger.warning(f"No valid ASR result for {audio_name}, skipping.")
        return None, None

    logger.info("Step 5: MOS Prediction & Filtering")
    avg_mos, mos_list = mos_prediction(audio, asr_result)
    logger.info(f"  Average MOS: {avg_mos}")
    filtered_list, all_list = filter_segments(mos_list)

    with open(os.path.join(save_path, "filtered_segments.json"), "w", encoding='utf8') as f:
        json.dump(filtered_list, f, indent=2, ensure_ascii=False)

    final_path = os.path.join(save_path, "all_segments.json")
    with open(final_path, "w", encoding='utf8') as f:
        json.dump(all_list, f, indent=2, ensure_ascii=False)

    logger.info("Step 6: Exporting to MP3")
    export_to_mp3(audio, all_list, save_path, audio_name)

    logger.info(f"All done, Saved to: {final_path}")
    return final_path, all_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to a single audio file to process.")
    parser.add_argument("--config_path", type=str, default="Emilia/config.json", help="Config path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store all processed outputs")
    parser.add_argument("--batch_size", type=int, default=16, help="ASR batch size")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type for Whisper model")
    parser.add_argument("--whisper_arch", type=str, default="medium", help="Whisper model size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ASR")
    
    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        args.compute_type = "int8"

    hf_token = os.getenv("HF_TOKEN")
    if cfg.get("huggingface_token") == "READ_FROM_ENVIRONMENT_VARIABLE" and hf_token:
        cfg["huggingface_token"] = hf_token
    
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("Hugging Face token is missing or invalid.")

    # Load Models
    logger.info("Loading models...")
    dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=cfg["huggingface_token"])
    dia_pipeline.to(device)
    
    asr_model = whisper_asr.load_asr_model(args.whisper_arch, device_name, compute_type=args.compute_type, threads=args.threads)
    vad = silero_vad.SileroVAD(device=device)
    separate_predictor1 = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)
    dnsmos_compute_score = dnsmos.ComputeScore(cfg["mos_model"]["primary_model_path"], device_name)
    logger.info("All models loaded.")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]

    main_process(args.input_file_path, save_path=args.output_dir)

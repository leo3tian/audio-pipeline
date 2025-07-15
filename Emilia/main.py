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
import logging
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import pandas as pd
## REFACTOR: Import TypedDict for creating a structured dictionary for our models.
from typing import TypedDict

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

## REFACTOR: Define a class to act as a type hint for the dictionary containing our models.
## This makes the code cleaner and easier to understand.
class ModelPack(TypedDict):
    separator: separate_fast.Predictor
    diarizer: Pipeline
    vad: silero_vad.SileroVAD
    asr: whisper_asr.WhisperASR
    dnsmos: dnsmos.ComputeScore

# The logger is initialized globally so it can be accessed by all functions.
logger = Logger.get_logger()

@time_logger
## REFACTOR: The 'cfg' dictionary is now passed as an argument.
def standardization(audio, cfg):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object.
        cfg (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, name, and sample rate.
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

    # Convert the audio file to WAV format using settings from the config
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
    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }

@time_logger
## REFACTOR: The source separation model ('predictor') is now passed as an argument.
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.
    This version processes the audio in chunks to avoid memory errors.
    """
    CHUNK_DURATION_SECONDS = 3600
    TARGET_SR = 44100
    original_sr = audio["sample_rate"]
    original_waveform = audio["waveform"]

    chunk_size_frames_orig = CHUNK_DURATION_SECONDS * original_sr
    final_vocals_chunks = []

    logger.info(f"Processing source separation in {CHUNK_DURATION_SECONDS}s chunks...")

    for i in tqdm.tqdm(range(0, len(original_waveform), chunk_size_frames_orig), desc="Separating Chunks"):
        original_chunk = original_waveform[i:i + chunk_size_frames_orig]
        resampled_chunk = librosa.resample(original_chunk, orig_sr=original_sr, target_sr=TARGET_SR)
        vocals_chunk_resampled, _ = predictor.predict(resampled_chunk)
        vocals_chunk_original_sr = librosa.resample(vocals_chunk_resampled[:, 0], orig_sr=TARGET_SR, target_sr=original_sr)
        final_vocals_chunks.append(vocals_chunk_original_sr)

    final_vocals = np.concatenate(final_vocals_chunks)
    audio["waveform"] = final_vocals
    logger.info("Source separation complete.")
    return audio

@time_logger
## REFACTOR: The diarization pipeline and device are now passed as arguments.
def speaker_diarization(audio, dia_pipeline, device):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        dia_pipeline (pyannote.audio.Pipeline): The pre-loaded speaker diarization model.
        device (torch.device): The device (CPU or CUDA) to run the model on.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    # Use the passed-in diarization model
    segments = dia_pipeline(
        {"waveform": waveform, "sample_rate": audio["sample_rate"], "channel": 0}
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    return diarize_df

@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels. (No changes needed here)
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

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]
    return filter_list

@time_logger
## REFACTOR: Pass the ASR model, batch size, and config dictionary as arguments.
def asr(vad_segments, audio, asr_model, batch_size, cfg):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.
        asr_model (whisper_asr.WhisperASR): The pre-loaded ASR model.
        batch_size (int): The batch size for transcription.
        cfg (dict): The configuration dictionary.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(vad_segments) == 0:
        return []

    # Get language settings from the passed-in config
    multilingual_flag = cfg["language"]["multilingual"]
    supported_languages = cfg["language"]["supported"]

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
        valid_vad_segments, valid_vad_segments_language = [], []
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        if len(valid_vad_segments) == 0: return []
        
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
                temp_audio, vad_segments_lang, batch_size=batch_size, language=language, print_progress=True
            )
            result = transcribe_result_temp["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        return sorted(all_transcribe_result, key=lambda x: x["start"])
    else:
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio, vad_segments, batch_size=batch_size, language=language, print_progress=True
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
## REFACTOR: Pass the DNSMOS model and config as arguments.
def mos_prediction(audio, vad_list, dnsmos_model, cfg):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.
    """
    audio_waveform = audio["waveform"]
    sample_rate = 16000

    audio_resampled = librosa.resample(
        audio_waveform, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio_resampled[start:end]
        
        # Use the passed-in DNSMOS model
        dnsmos_score = dnsmos_model(segment, sample_rate, False)["OVRL"]
        vad_list[index]["dnsmos"] = dnsmos_score

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list if "dnsmos" in vad])
    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")
    return predict_dnsmos, vad_list

def filter(mos_list):
    """
    Filter out the segments with MOS scores, wrong char duration, and total duration.
    (No changes needed here)
    """
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(mos_list)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    logger.debug(
        f"> {all_segment - filtered_segment}/{all_segment} {(all_segment - filtered_segment) / all_segment:.2%} segments filtered."
    )
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]
    all_list = [mos_list[idx] for idx, _ in all_audio_stats]
    return filtered_list, all_list


## REFACTOR: This is the main refactored function.
## It now accepts the pre-loaded models and other necessary configs,
## avoiding the need for global variables and enabling efficient reuse.
@time_logger
def main_process(
    audio_path: str,
    models: ModelPack,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    save_path: str = None,
    audio_name: str = None,
):
    """
    Process an audio file using pre-loaded models.

    Args:
        audio_path (str): Path to the audio file.
        models (ModelPack): A dictionary containing all the pre-loaded models.
        cfg (dict): The configuration dictionary.
        device (torch.device): The device (CPU or CUDA) to run models on.
        batch_size (int): The batch size for ASR.
        save_path (str, optional): Directory to save outputs.
        audio_name (str, optional): Name for the output files.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
        return None, None

    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    if save_path is None:
        save_path = os.path.join(os.path.dirname(audio_path) + "_processed", audio_name)
    os.makedirs(save_path, exist_ok=True)
    logger.debug(f"Processing audio: {audio_name}, save to: {save_path}")

    logger.info("Step 0: Standardization")
    audio = standardization(audio_path, cfg)

    logger.info("Step 1: Source Separation")
    ## REFACTOR: Pass the separator model from the 'models' dictionary.
    audio = source_separation(models["separator"], audio)

    logger.info("Step 2: Speaker Diarization")
    ## REFACTOR: Pass the diarizer model and device.
    speakerdia = speaker_diarization(audio, models["diarizer"], device)

    logger.info("Step 3: Fine-grained Segmentation by VAD")
    ## REFACTOR: Call the 'vad' method on the VAD model from the 'models' dictionary.
    vad_list = models["vad"].vad(speakerdia, audio)
    segment_list = cut_by_speaker_label(vad_list)

    logger.info("Step 4: ASR")
    ## REFACTOR: Pass the ASR model, batch size, and config.
    asr_result = asr(segment_list, audio, models["asr"], batch_size, cfg)
    if not asr_result:
        logger.warning(f"No valid ASR result for {audio_name}, skipping.")
        return None, None

    logger.info("Step 5: Filter")
    logger.info("Step 5.1: MOS Prediction")
    ## REFACTOR: Pass the DNSMOS model and config.
    avg_mos, mos_list = mos_prediction(audio, asr_result, models["dnsmos"], cfg)
    logger.info(f"Step 5.1: done, average MOS: {avg_mos}")

    logger.info("MODIFIED Step 5.2: Filter out files with less than average MOS")
    filtered_list, all_list = filter(mos_list)

    with open(os.path.join(save_path, "filtered_segments.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_list, f, indent=2, ensure_ascii=False)

    final_path = os.path.join(save_path, "all_segments.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_list, f, indent=2, ensure_ascii=False)

    logger.info("Step 6: Write result into MP3 and JSON file")
    export_to_mp3(audio, all_list, save_path, audio_name)

    logger.info(f"All done, Saved to: {final_path}")
    return final_path, all_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", type=str, default=None)
    parser.add_argument("--input_file_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--whisper_arch", type=str, default="medium")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exit_pipeline", type=bool, default=False)
    args = parser.parse_args()

    # Set up logger
    if args.quiet:
        logger.setLevel(logging.WARNING)

    cfg = load_cfg(args.config_path)
    cfg["huggingface_token"] = os.getenv("HF_TOKEN")

    if args.input_folder_path and not args.input_file_path:
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.info("Loading models for standalone execution...")

    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        args.compute_type = "int8"

    check_env(logger)

    ## REFACTOR: This section remains for when you run main.py directly.
    ## The models are loaded here, then passed to main_process.
    
    # 1. Load Speaker Diarization Model
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"] or not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("Hugging Face token is missing or invalid.")
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=cfg["huggingface_token"]
    )
    dia_pipeline.to(device)

    # 2. Load ASR Model
    logger.debug(" * Loading ASR Model")
    asr_model = whisper_asr.load_asr_model(
        args.whisper_arch,
        device_name,
        compute_type=args.compute_type,
        threads=args.threads,
        asr_options={
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
        },
    )

    # 3. Load VAD Model
    logger.debug(" * Loading VAD Model")
    vad_model = silero_vad.SileroVAD(device=device)

    # 4. Load Background Noise Separation Model
    logger.debug(" * Loading Background Noise Model")
    separator_model = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)

    # 5. Load DNSMOS Scoring Model
    logger.debug(" * Loading DNSMOS Model")
    dnsmos_model = dnsmos.ComputeScore(cfg["mos_model"]["primary_model_path"], device_name)
    logger.debug("All models loaded")

    ## REFACTOR: Pack all loaded models into the 'ModelPack' dictionary.
    models: ModelPack = {
        "separator": separator_model,
        "diarizer": dia_pipeline,
        "vad": vad_model,
        "asr": asr_model,
        "dnsmos": dnsmos_model,
    }

    if args.input_file_path:
        if not os.path.exists(args.input_file_path):
             raise FileNotFoundError(f"Input file not found: {args.input_file_path}")
        logger.info(f"Processing single file: {args.input_file_path}")
        
        ## REFACTOR: Call main_process with the new arguments.
        main_process(
            audio_path=args.input_file_path,
            models=models,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
            save_path=args.output_dir
        )

    elif args.input_folder_path:
        input_folder_path = args.input_folder_path
        if not os.path.exists(input_folder_path):
            raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

        audio_paths = get_audio_files(input_folder_path)
        logger.info(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

        for path in audio_paths:
            audio_name = os.path.splitext(os.path.basename(path))[0]
            specific_save_path = os.path.join(args.output_dir, audio_name) if args.output_dir else None
            
            ## REFACTOR: Call main_process with the new arguments for each file.
            main_process(
                audio_path=path,
                models=models,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                save_path=specific_save_path
            )
    else:
        logger.error("No input specified. Please provide --input_file_path or --input_folder_path.")

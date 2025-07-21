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

# Define a mock class that safely mimics tqdm's behavior but does nothing.
class _TqdmMock:
    """A mock class to replace tqdm and suppress all output."""
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        # Allow the object to be used in for loops
        return iter(self.iterable if self.iterable is not None else [])

    # This is the key: a catch-all for any method calls
    def __getattr__(self, *args, **kwargs):
        """Catch any method call (.set_description, .update, etc.) and do nothing."""
        return lambda *args, **kwargs: None

# Replace the real tqdm with our silent mock class
tqdm.tqdm = _TqdmMock

import warnings
import logging
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import pandas as pd
from typing import TypedDict
import threading
from queue import Queue
import soxr
## FIX: Changed from multiprocessing to ThreadPoolExecutor for this specific task
from concurrent.futures import ThreadPoolExecutor

from utils.tool import (
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

class ModelPack(TypedDict):
    separator: separate_fast.Predictor
    diarizer: Pipeline
    vad: silero_vad.SileroVAD
    asr: object
    dnsmos: dnsmos.ComputeScore

# The logger is initialized globally so it can be accessed by all functions.
logger = Logger.get_logger()
logger.setLevel(logging.WARNING)


# --- OPTIMIZED I/O FUNCTIONS ---

def _export_mp3_segment_thread(args):
    """Helper function for ThreadPoolExecutor to export a single MP3."""
    segment, audio_segment, save_path, audio_name = args
    start_ms = int(segment["start"] * 1000)
    end_ms = int(segment["end"] * 1000)
    
    # Extract the audio chunk
    chunk = audio_segment[start_ms:end_ms]
    
    # Define the output filename
    idx = segment.get("segment_id", 0) # Use a segment ID if available
    output_filename = os.path.join(save_path, f"{audio_name}_{idx:06d}.mp3")
    
    # Export the chunk to MP3
    try:
        chunk.export(output_filename, format="mp3")
    except Exception as e:
        logger.error(f"Failed to export segment {idx} to {output_filename}: {e}")
    return True

@time_logger
def export_to_mp3(audio, segments, save_path, audio_name):
    """
    Exports audio segments to MP3 files in parallel using a ThreadPoolExecutor.
    """
    if not segments:
        logger.info("No segments to export.")
        return

    logger.info(f"Starting parallel export of {len(segments)} MP3 files...")
    
    # Convert numpy waveform to pydub AudioSegment once
    waveform_int16 = (audio["waveform"] * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        waveform_int16.tobytes(), 
        frame_rate=audio["sample_rate"],
        sample_width=waveform_int16.dtype.itemsize,
        channels=1
    )
    
    # Prepare arguments for each worker
    tasks = []
    for i, segment in enumerate(segments):
        segment['segment_id'] = i # Add an index for unique filenames
        tasks.append((segment, audio_segment, save_path, audio_name))

    # For many small, fast tasks that involve I/O (writing to disk),
    # a ThreadPoolExecutor is more efficient due to lower overhead.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm.tqdm(executor.map(_export_mp3_segment_thread, tasks), total=len(tasks), desc="Exporting to MP3"))

    logger.info("Parallel MP3 export complete.")


# --- CORE PROCESSING FUNCTIONS ---

@time_logger
def standardization(audio, cfg):
    """
    Preprocess the audio file.
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
    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }

@time_logger
def source_separation(predictor, audio):
    """
    Separate audio into vocals and non-vocals using a three-stage pipeline.
    """
    CHUNK_DURATION_SECONDS = 600 # 10 minutes
    TARGET_SR = 44100
    original_sr = audio["sample_rate"]
    original_waveform = audio["waveform"]
    chunk_size_frames_orig = CHUNK_DURATION_SECONDS * original_sr
    
    to_gpu_queue = Queue(maxsize=2)
    from_gpu_queue = Queue(maxsize=2)
    final_vocals_chunks = []
    
    def producer():
        for i in range(0, len(original_waveform), chunk_size_frames_orig):
            original_chunk = original_waveform[i:i + chunk_size_frames_orig]
            resampled_chunk = soxr.resample(original_chunk, original_sr, TARGET_SR, quality='hq')
            to_gpu_queue.put(resampled_chunk)
        to_gpu_queue.put(None)

    def post_processor():
        while True:
            predicted_chunk = from_gpu_queue.get()
            if predicted_chunk is None:
                break
            vocals_chunk_original_sr = soxr.resample(predicted_chunk[:, 0], TARGET_SR, original_sr, quality='hq')
            final_vocals_chunks.append(vocals_chunk_original_sr)

    producer_thread = threading.Thread(target=producer)
    post_processor_thread = threading.Thread(target=post_processor)
    producer_thread.start()
    post_processor_thread.start()

    num_chunks = (len(original_waveform) + chunk_size_frames_orig - 1) // chunk_size_frames_orig
    logger.info(f"Fully optimized source separation started for {num_chunks} chunks...")
    
    for _ in tqdm.tqdm(range(num_chunks), desc="Separating Chunks"):
        resampled_chunk = to_gpu_queue.get()
        if resampled_chunk is None:
            break
        vocals_chunk_resampled, _ = predictor.predict(resampled_chunk)
        from_gpu_queue.put(vocals_chunk_resampled)

    from_gpu_queue.put(None)
    producer_thread.join()
    post_processor_thread.join()

    final_vocals = np.concatenate(final_vocals_chunks)
    audio["waveform"] = final_vocals
    logger.info("Source separation complete.")
    return audio


@time_logger
def speaker_diarization(audio, dia_pipeline, device):
    """Perform speaker diarization."""
    logger.debug(f"Start speaker diarization")
    
    ## MEMORY LEAK FIX: Create and destroy the tensor within this function's scope.
    waveform_tensor = torch.from_numpy(audio["waveform"]).to(device)
    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    segments = dia_pipeline({"waveform": waveform_tensor, "sample_rate": audio["sample_rate"]})

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    if not diarize_df.empty:
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        
    ## MEMORY LEAK FIX: Explicitly delete the large tensor to free VRAM.
    del waveform_tensor
    
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

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]
    return filter_list

@time_logger
def asr(vad_segments, audio, asr_model, batch_size, cfg):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments.
    """
    if len(vad_segments) == 0:
        return []

    multilingual_flag = cfg["language"]["multilingual"]
    supported_languages = cfg["language"]["supported"]

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]

    for idx, segment in enumerate(vad_segments):
        segment["start"] -= start_time
        segment["end"] -= start_time
    
    temp_audio = soxr.resample(
        temp_audio, audio["sample_rate"], 16000, quality='hq'
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
def mos_prediction(audio, vad_list, dnsmos_model, cfg):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.
    """
    audio_waveform = audio["waveform"]
    sample_rate = 16000

    audio_resampled = soxr.resample(
        audio_waveform, cfg["entrypoint"]["SAMPLE_RATE"], sample_rate, quality='hq'
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio_resampled[start:end]
        
        dnsmos_score = dnsmos_model(segment, sample_rate, False)["OVRL"]
        vad_list[index]["dnsmos"] = dnsmos_score

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list if "dnsmos" in vad])
    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")
    return predict_dnsmos, vad_list

def filter(mos_list):
    """
    Filter out the segments with MOS scores, wrong char duration, and total duration.
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
    audio = source_separation(models["separator"], audio)

    logger.info("Step 2: Speaker Diarization")
    speakerdia = speaker_diarization(audio, models["diarizer"], device)

    logger.info("Step 3: Fine-grained Segmentation by VAD")
    vad_list = models["vad"].vad(speakerdia, audio)
    segment_list = cut_by_speaker_label(vad_list)

    logger.info("Step 4: ASR")
    asr_result = asr(segment_list, audio, models["asr"], batch_size, cfg)
    if not asr_result:
        logger.warning(f"No valid ASR result for {audio_name}, skipping.")
        return None, None

    logger.info("Step 5: Filter")
    logger.info("Step 5.1: MOS Prediction")
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
    
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"] or not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("Hugging Face token is missing or invalid.")
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=cfg["huggingface_token"]
    )
    dia_pipeline.to(device)

    logger.debug(" * Loading ASR Model")
    asr_model = whisper_asr.load_asr_model(
        args.whisper_arch, device_name, compute_type=args.compute_type, threads=args.threads,
        asr_options=cfg.get("asr")
    )

    logger.debug(" * Loading VAD Model")
    vad_model = silero_vad.SileroVAD(device=device)

    logger.debug(" * Loading Background Noise Separation Model")
    separator_model = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)

    logger.debug(" * Loading DNSMOS Scoring Model")
    dnsmos_model = dnsmos.ComputeScore(cfg["mos_model"]["primary_model_path"], device_name)
    logger.debug("All models loaded")

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

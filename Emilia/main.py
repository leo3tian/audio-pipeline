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

# These imports are now relative for when worker.py imports this file
from .utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from .utils.logger import Logger, time_logger
from .models import separate_fast, dnsmos, whisper_asr, silero_vad

warnings.filterwarnings("ignore")

# This will be populated by the load_models function
models_container = {}
# This will hold the global configuration
cfg = {}


@time_logger
def standardization(audio):
    """Preprocess the audio file."""
    if isinstance(audio, str):
        audio = AudioSegment.from_file(audio)
    
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)

    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude

    return {"waveform": waveform, "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"]}


@time_logger
def source_separation(audio):
    """Separate vocals using a pre-loaded model."""
    predictor = models_container['source_separator']
    rate = audio["sample_rate"]
    mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)
    vocals, _ = predictor.predict(mix)
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    audio["waveform"] = vocals[:, 0]
    return audio


@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization. Reverted to the original, faster implementation
    as the 'load models once' architecture should solve the memory issues.
    """
    dia_pipeline = models_container['diarization']
    device = models_container['device']
    
    waveform = torch.tensor(audio["waveform"]).to(device).unsqueeze(0)
    
    segments = dia_pipeline({"waveform": waveform, "sample_rate": audio["sample_rate"]})

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    return diarize_df


# ... (Other processing functions like cut_by_speaker_label, asr, mos_prediction, filter would be here)
# These functions would also be modified to take `models` as an argument if they need a model.


def main_process(audio_path, save_path, audio_name):
    """
    Main processing pipeline for a single audio file, using pre-loaded global models.
    """
    logger = Logger.get_logger()
    
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Processing audio: {audio_name}")
    
    audio = standardization(audio_path)
    audio = source_separation(audio)
    speaker_df = speaker_diarization(audio)
    
    # Placeholder for the rest of the pipeline (VAD, ASR, etc.)
    # For now, we just confirm the first few steps work.
    # We will save the diarization result to confirm success.
    output_csv_path = os.path.join(save_path, f"{audio_name}_diarization.csv")
    speaker_df.to_csv(output_csv_path, index=False)
    
    logger.info(f"✅ Finished processing {audio_name}. Results saved to {save_path}")


def load_all_models(args):
    """
    Loads all necessary models into memory and populates the global models_container.
    This function should be called only once per worker process.
    """
    global models_container, cfg
    
    cfg = load_cfg(args.config_path)
    
    print(f"--- Loading all models into memory for device {args.gpu_id}... ---")
    
    device = torch.device(f"cuda:{args.gpu_id}")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set.")

    models_container = {
        'diarization': Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device),
        'source_separator': separate_fast.Predictor(args=cfg["separate"]["step1"], device=f"cuda:{args.gpu_id}"),
        # Add other models here as needed (ASR, VAD, DNSMOS)
        'device': device,
    }
    print(f"--- All models loaded successfully for device {args.gpu_id}. ---")


if __name__ == "__main__":
    # This block allows the script to be run standalone for testing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="Emilia/config.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use.")
    
    args = parser.parse_args()
    
    # Set the visible device for this standalone run
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    logger = Logger.get_logger()
    
    # Load models once
    load_all_models(args)
    
    # Process the single file
    audio_name = os.path.splitext(os.path.basename(args.input_file_path))[0]
    save_path = os.path.join(args.output_dir, audio_name)
    main_process(args.input_file_path, save_path, audio_name)

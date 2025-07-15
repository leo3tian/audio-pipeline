import os
import json
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import time
import random
import torch
import sys

from pyannote.audio import Pipeline

emilia_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Emilia'))
if emilia_path not in sys.path:
    sys.path.insert(0, emilia_path)

from Emilia.main import main_process, ModelPack # Our refactored pipeline function
from Emilia.models import separate_fast, dnsmos, whisper_asr, silero_vad
from Emilia.utils.tool import load_cfg

# --- Configuration ---
S3_BUCKET = os.environ.get("WORKER_BUCKET") # "sptfy-dataset" # yt-pipeline-bucket
S3_RAW_AUDIO_PREFIX = "raw-audio/" # raw_audio for yt
S3_PROCESSED_PREFIX = "processed/"
S3_TASKS_BASE_PREFIX = "tasks/"
MAX_EMILIA_WORKERS = int(os.environ.get("MAX_EMILIA_WORKERS", 9999))
# EMILIA_PIPE_PATH is no longer needed as we call the function directly.
EMILIA_CONFIG_PATH = "Emilia/config.json"

# --- Emilia Config ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 12))
## REFACTOR: Define ASR model parameters here for the worker to use.
WHISPER_ARCH = "medium"
COMPUTE_TYPE = "float16"
CPU_THREADS = 4


def claim_processing_task(s3_client):
    """
    Atomically claims a task by listing and then attempting to delete it.
    The first worker to successfully delete the task from 'todo' wins the claim.
    """
    todo_prefix = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_todo/')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=todo_prefix, MaxKeys=10)

    for page in pages:
        if 'Contents' not in page:
            continue

        tasks = page.get('Contents', [])
        random.shuffle(tasks)

        for obj in tasks:
            task_key = obj['Key']
            if not task_key.endswith('.task'):
                continue

            try:
                task_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=task_key)
                video_id = task_obj['Body'].read().decode('utf-8')

                s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)

                in_progress_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_in_progress', f"{video_id}.task")
                s3_client.put_object(Bucket=S3_BUCKET, Key=in_progress_key, Body=video_id)

                return {'key': in_progress_key, 'video_id': video_id}

            except ClientError as e:
                if e.response['Error']['Code'] in ['NoSuchKey', '404']:
                    continue
                else:
                    raise
    return None

def complete_processing_task(s3_client, task_key):
    """
    Moves a completed processing task file to the 'processing_completed' directory.
    """
    video_id = os.path.basename(task_key).replace('.task', '')
    completed_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_completed', f"{video_id}.task")
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=task_key)
        s3_client.copy_object(
            Bucket=S3_BUCKET,
            CopySource={'Bucket': S3_BUCKET, 'Key': task_key},
            Key=completed_key
        )
        s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  Note: Task for {video_id} was already completed by another worker.")
        else:
            print(f"  Warning: Unexpected S3 error while completing task {task_key}: {e}")
            raise

## REFACTOR: The run_emilia_pipe function is no longer needed because we are
## calling the main_process function directly, avoiding the slow subprocess overhead.
# def run_emilia_pipe(...):
#     ...

def processing_worker(rank: int, device: str):
    """
    A worker process that loads models once, then continuously claims and processes audio files.
    """
    s3_client = boto3.client('s3')
    print(f"GPU-{device} (Rank {rank}): Starting...")

    # --------------------------------------------------------------------------
    ## REFACTOR: LOAD MODELS ONCE PER WORKER
    # This entire block loads the AI models into memory *before* the task loop starts.
    # This is the core of the performance optimization.
    # --------------------------------------------------------------------------
    print(f"GPU-{device} (Rank {rank}): Loading all models into memory...")
    
    torch_device = torch.device(f"cuda:{device}")
    device_name = f"cuda:{device}"

    # Load config file
    cfg = load_cfg(EMILIA_CONFIG_PATH)
    cfg["huggingface_token"] = os.getenv("HF_TOKEN")

    # 1. Load Speaker Diarization Model
    if not cfg["huggingface_token"] or not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("Hugging Face token is missing or invalid.")
    diarizer_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=cfg["huggingface_token"]
    )
    diarizer_model.to(torch_device)

    # 2. Load ASR Model
    asr_model = whisper_asr.load_asr_model(
        whisper_arch=WHISPER_ARCH,
        device="cuda",
        device_index=int(device),
        compute_type=COMPUTE_TYPE,
        threads=CPU_THREADS,
        asr_options=cfg.get("asr") # Pass ASR options from config if they exist
    )

    # 3. Load VAD Model
    vad_model = silero_vad.SileroVAD(device=torch_device)

    # 4. Load Background Noise Separation Model
    separator_model = separate_fast.Predictor(args=cfg["separate"]["step1"], device="cuda")

    # 5. Load DNSMOS Scoring Model
    dnsmos_model = dnsmos.ComputeScore(cfg["mos_model"]["primary_model_path"], "cuda")
    
    # Pack all models into the structured dictionary for easy passing
    models: ModelPack = {
        "separator": separator_model,
        "diarizer": diarizer_model,
        "vad": vad_model,
        "asr": asr_model,
        "dnsmos": dnsmos_model,
    }
    print(f"GPU-{device} (Rank {rank}): All models loaded successfully.")
    # --------------------------------------------------------------------------
    # END OF MODEL LOADING BLOCK
    # --------------------------------------------------------------------------

    while True:
        task = claim_processing_task(s3_client)
        if not task:
            print(f"GPU-{device} (Rank {rank}): No more tasks found. Exiting.")
            break

        video_id = task['video_id']
        print(f"GPU-{device} (Rank {rank}): Claimed task for video: {video_id}")

        try:
            with tempfile.TemporaryDirectory(prefix=f"gpu_worker_{device}_") as temp_dir:
                # 1. Find and download the audio file from S3
                s3_search_prefix = f"{S3_RAW_AUDIO_PREFIX}{video_id}"
                print(f"  GPU-{device}: Searching for audio file with prefix '{s3_search_prefix}'...")
                
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_search_prefix, MaxKeys=1)
                if 'Contents' not in response or not response['Contents']:
                    raise FileNotFoundError(f"Could not find any audio file for video_id: {video_id}")

                s3_audio_key = response['Contents'][0]['Key']
                local_filename = os.path.basename(s3_audio_key)
                local_audio_path = Path(temp_dir) / local_filename
                print(f"  GPU-{device}: Found and downloading {s3_audio_key}...")
                s3_client.download_file(S3_BUCKET, s3_audio_key, str(local_audio_path))

                # --------------------------------------------------------------------------
                ## REFACTOR: DIRECTLY CALL THE PROCESSING PIPELINE
                # Instead of a slow subprocess, we now call the Python function directly,
                # passing the pre-loaded models for maximum efficiency.
                # --------------------------------------------------------------------------
                emilia_output_dir = Path(temp_dir) / "processed"
                os.makedirs(emilia_output_dir, exist_ok=True)
                print(f"  GPU-{device}: Starting Emilia pipe for {video_id}...")
                
                main_process(
                    audio_path=str(local_audio_path),
                    models=models,
                    cfg=cfg,
                    device=torch_device,
                    batch_size=BATCH_SIZE,
                    save_path=str(emilia_output_dir),
                    audio_name=video_id
                )
                # --------------------------------------------------------------------------

                # 4. Upload the processed results
                s3_processed_prefix = f"{S3_PROCESSED_PREFIX}{video_id}"
                print(f"  GPU-{device}: Uploading processed results for {video_id}...")
                upload_directory_to_s3(emilia_output_dir, S3_BUCKET, s3_processed_prefix)
            
            # 5. Mark the task as complete
            complete_processing_task(s3_client, task['key'])
            print(f"  GPU-{device}: ✅ Finished and completed task for video: {video_id}")

        except Exception as e:
            # Basic error handling. Consider moving the task to a 'failed' queue here.
            print(f"  GPU-{device}: [!!!] CRITICAL FAILURE on video {video_id}. Error: {e}")
            # In a production system, you'd want to move the task to a failed state
            # instead of just sleeping.
            time.sleep(10)

def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix."""
    s3_client = boto3.client("s3")
    for local_file in local_directory.rglob("*"):
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        # Fallback for systems without nvidia-smi or GPUs
        return []

def main():
    """Orchestrates the pool of GPU worker processes."""
    processes = []
    available_devices = get_available_gpus()[:MAX_EMILIA_WORKERS]
    world_size = len(available_devices)

    if world_size == 0:
        print("[!] No GPUs detected. Aborting.")
        return
        
    print(f"🚀 Starting {world_size} GPU worker processes...")
    
    for rank, device in enumerate(available_devices):
        p = multiprocessing.Process(target=processing_worker, args=(rank, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ All GPU workers have finished.")

if __name__ == "__main__":
    # 'spawn' is a safer start method for multiprocessing with CUDA
    multiprocessing.set_start_method("spawn", force=True)
    main()

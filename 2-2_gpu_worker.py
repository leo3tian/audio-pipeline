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
import sys
from concurrent.futures import ThreadPoolExecutor
import tqdm

# --- Configuration ---
S3_BUCKET = os.environ.get("WORKER_BUCKET")
if S3_BUCKET is None:
    raise ValueError("FATAL: The environment variable 'WORKER_BUCKET' is not set. Please set it to your S3 bucket name.")

S3_RAW_AUDIO_PREFIX = "raw-audio/"
S3_PROCESSED_PREFIX = "processed/"
S3_TASKS_BASE_PREFIX = "tasks/"
EMILIA_CONFIG_PATH = "Emilia/config.json"
WORKERS_PER_GPU = int(os.environ.get("WORKERS_PER_GPU", 1))
MAX_GPUS = int(os.environ.get("MAX_GPUS", 999))

# --- Emilia Config ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 12))
WHISPER_ARCH = "medium"
COMPUTE_TYPE = "float16"
CPU_THREADS = 4
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", None)


def claim_processing_task(s3_client):
    """Atomically claims a task by listing and then attempting to delete it."""
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
    """Moves a completed processing task file to the 'processing_completed' directory."""
    video_id = os.path.basename(task_key).replace('.task', '')
    completed_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_completed', f"{video_id}.task")
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=task_key)
        s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': task_key}, Key=completed_key)
        s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  Note: Task for {video_id} was already completed by another worker.")
        else:
            print(f"  Warning: Unexpected S3 error while completing task {task_key}: {e}")
            raise

def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix in parallel."""
    s3_client = boto3.client("s3")
    files_to_upload = [f for f in local_directory.rglob("*") if f.is_file()]
    if not files_to_upload:
        return
    print(f"  Starting parallel upload of {len(files_to_upload)} files to s3://{s3_bucket}/{s3_prefix}")
    def _upload_file(local_file_path):
        s3_key = f"{s3_prefix}/{local_file_path.relative_to(local_directory)}"
        try:
            s3_client.upload_file(str(local_file_path), s3_bucket, s3_key)
        except Exception as e:
            print(f"  [!] Failed to upload {local_file_path}: {e}")
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(tqdm.tqdm(executor.map(_upload_file, files_to_upload), total=len(files_to_upload), desc="Uploading to S3"))

def processing_worker(rank: int, assigned_gpu_id: str):
    """
    The main worker function. It's now called by a launcher to ensure GPU isolation.
    """
    import torch
    from pyannote.audio import Pipeline
    
    emilia_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Emilia'))
    if emilia_path not in sys.path:
        sys.path.insert(0, emilia_path)
        
    from main import main_process, ModelPack
    from models import separate_fast, dnsmos, whisper_asr, silero_vad
    from utils.tool import load_cfg

    device_id_for_process = 0 
    
    s3_client = boto3.client('s3')
    print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Starting...")

    print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Loading all models into memory...")
    
    torch_device = torch.device(f"cuda:{device_id_for_process}")
    generic_cuda_device = "cuda"
    
    cfg = load_cfg(EMILIA_CONFIG_PATH)
    cfg["huggingface_token"] = os.getenv("HF_TOKEN")

    diarizer_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=cfg["huggingface_token"])
    diarizer_model.to(torch_device)

    asr_model = whisper_asr.load_asr_model(
        whisper_arch=WHISPER_ARCH, device=generic_cuda_device, device_index=device_id_for_process,
        compute_type=COMPUTE_TYPE, threads=CPU_THREADS, asr_options=cfg.get("asr"), language=ASR_LANGUAGE
    )
    if ASR_LANGUAGE:
        print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: ASR model loaded for language: {ASR_LANGUAGE}")

    vad_model = silero_vad.SileroVAD(device=torch_device)
    separator_model = separate_fast.Predictor(
        args=cfg["separate"]["step1"], 
        device=generic_cuda_device, 
        device_index=device_id_for_process
    )
    dnsmos_model = dnsmos.ComputeScore(
        cfg["mos_model"]["primary_model_path"], 
        device=generic_cuda_device,
        device_index=device_id_for_process
    )
    
    models: ModelPack = {
        "separator": separator_model, "diarizer": diarizer_model, "vad": vad_model,
        "asr": asr_model, "dnsmos": dnsmos_model,
    }
    print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: All models loaded successfully.")

    while True:
        task = claim_processing_task(s3_client)
        if not task:
            break
        video_id = task['video_id']
        print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Claimed task for video: {video_id}")
        try:
            with tempfile.TemporaryDirectory(prefix=f"gpu_worker_{assigned_gpu_id}_{rank}_") as temp_dir:
                s3_search_prefix = f"{S3_RAW_AUDIO_PREFIX}{video_id}"
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_search_prefix, MaxKeys=1)
                if 'Contents' not in response or not response['Contents']:
                    raise FileNotFoundError(f"Could not find any audio file for video_id: {video_id}")
                s3_audio_key = response['Contents'][0]['Key']
                local_filename = os.path.basename(s3_audio_key)
                local_audio_path = Path(temp_dir) / local_filename
                s3_client.download_file(S3_BUCKET, s3_audio_key, str(local_audio_path))
                emilia_output_dir = Path(temp_dir) / "processed"
                os.makedirs(emilia_output_dir, exist_ok=True)
                main_process(
                    audio_path=str(local_audio_path), models=models, cfg=cfg,
                    device=torch_device, batch_size=BATCH_SIZE,
                    save_path=str(emilia_output_dir), audio_name=video_id
                )
                s3_processed_prefix = f"{S3_PROCESSED_PREFIX}{video_id}"
                upload_directory_to_s3(emilia_output_dir, S3_BUCKET, s3_processed_prefix)
            
            complete_processing_task(s3_client, task['key'])
            print(f"  Worker-{rank} on Physical-GPU-{assigned_gpu_id}: ✅ Finished and completed task for video: {video_id}")

        except Exception as e:
            print(f"  Worker-{rank} on Physical-GPU-{assigned_gpu_id}: [!!!] CRITICAL FAILURE on video {video_id}. Error: {e}")
            time.sleep(10)
        finally:
            torch.cuda.empty_cache()


def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        return []

def worker_launcher(rank: int, assigned_gpu_id: str):
    """Sets the environment for a worker and then calls the main worker function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu_id
    processing_worker(rank, assigned_gpu_id)

def main():
    """Orchestrates the pool of GPU worker processes."""
    processes = []
    all_gpus = get_available_gpus()
    max_gpus_to_use = int(os.environ.get("MAX_GPUS", 999))
    available_gpus = all_gpus[:max_gpus_to_use]
    
    if not available_gpus:
        print("[!] No GPUs detected or all GPUs were excluded by MAX_GPUS. Aborting.")
        return

    worker_assignments = (available_gpus * WORKERS_PER_GPU)
    total_workers = len(worker_assignments)
    print(f"🚀 Starting {total_workers} worker processes across {len(available_gpus)} GPUs ({WORKERS_PER_GPU} workers per GPU)...")
    
    for rank, device_id in enumerate(worker_assignments):
        p = multiprocessing.Process(target=worker_launcher, args=(rank, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ All GPU workers have finished.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

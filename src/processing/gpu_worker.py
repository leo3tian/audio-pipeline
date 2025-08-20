import os
import json
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor
import tqdm

# --- Configuration ---
S3_BUCKET = os.environ.get("R2_BUCKET")
if S3_BUCKET is None:
    raise ValueError("FATAL: The environment variable 'R2_BUCKET' is not set. Please set it to your R2 bucket name.")

S3_RAW_AUDIO_PREFIX = os.environ.get("RAW_AUDIO_PREFIX", "raw_audio/")
S3_PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
S3_TASKS_BASE_PREFIX = os.environ.get("TASKS_BASE_PREFIX", "tasks/")
# Resolve Emilia path from repo root to avoid CWD issues
REPO_ROOT = Path(__file__).resolve().parents[2]
EMILIA_DIR = REPO_ROOT / "Emilia"
EMILIA_CONFIG_PATH = str(EMILIA_DIR / "config.json")
WORKERS_PER_GPU = int(os.environ.get("WORKERS_PER_GPU", 1))
MAX_GPUS = int(os.environ.get("MAX_GPUS", 999))

# Allow configuring model devices and separation chunk size
SEPARATION_DEVICE = os.environ.get("SEPARATION_DEVICE", "cuda")
DNSMOS_DEVICE = os.environ.get("DNSMOS_DEVICE", "cuda")
SEPARATION_CHUNKS_ENV = os.environ.get("SEPARATION_CHUNKS")

# --- Emilia Config ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 12))
WHISPER_ARCH = "medium"
COMPUTE_TYPE = "float16"
CPU_THREADS = 4
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", None)


def create_r2_client():
    """Create an S3-compatible client for Cloudflare R2 only."""
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not endpoint_url or not access_key or not secret_key:
        raise ValueError("R2 configuration missing. Set R2_ENDPOINT_URL (or R2_ENDPOINT), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.")
    base_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        region_name="auto",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=base_config,
    )


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
                body = task_obj['Body'].read().decode('utf-8')
                task_payload = json.loads(body)
                episode_id = task_payload.get("episode_id")
                language = task_payload.get("language")
                audio_key = task_payload.get("audio_key")
                # Fallback: derive language from audio_key if missing/empty
                if (not language) and audio_key:
                    segs = audio_key.strip('/').split('/')
                    if 'raw_audio' in segs:
                        idx = segs.index('raw_audio')
                        if idx + 1 < len(segs):
                            language = segs[idx + 1]
                            task_payload["language"] = language
                            body = json.dumps(task_payload)
                # move to in_progress
                s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
                in_progress_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_in_progress', f"{episode_id}.task")
                s3_client.put_object(Bucket=S3_BUCKET, Key=in_progress_key, Body=body)
                return {
                    'key': in_progress_key,
                    'episode_id': episode_id,
                    'language': language,
                    'audio_key': audio_key,
                }
            except ClientError as e:
                if e.response['Error']['Code'] in ['NoSuchKey', '404']:
                    continue
                else:
                    raise
    return None


def complete_processing_task(s3_client, task_key):
    """Moves a completed processing task file to the 'processing_completed' directory."""
    episode_id = os.path.basename(task_key).replace('.task', '')
    completed_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_completed', f"{episode_id}.task")
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=task_key)
        s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': task_key}, Key=completed_key)
        s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  Note: Task for {episode_id} was already completed by another worker.")
        else:
            print(f"  Warning: Unexpected R2 error while completing task {task_key}: {e}")
            raise


def upload_directory_to_r2(s3_client, local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific R2 prefix in parallel."""
    files_to_upload = [f for f in local_directory.rglob("*") if f.is_file()]
    if not files_to_upload:
        return
    print(f"  Starting parallel upload of {len(files_to_upload)} files to r2://{s3_bucket}/{s3_prefix}")
    def _upload_file(local_file_path):
        s3_key = f"{s3_prefix}/{local_file_path.relative_to(local_directory)}"
        try:
            s3_client.upload_file(str(local_file_path), s3_bucket, s3_key)
        except Exception as e:
            print(f"  [!] Failed to upload {local_file_path}: {e}")
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(tqdm.tqdm(executor.map(_upload_file, files_to_upload), total=len(files_to_upload), desc="Uploading to R2"))


def processing_worker(rank: int, assigned_gpu_id: str):
    """
    The main worker function. It's now called by a launcher to ensure GPU isolation.
    """
    import torch
    from pyannote.audio import Pipeline
    
    emilia_path = os.environ.get("EMILIA_PATH") or str(EMILIA_DIR)
    if emilia_path not in sys.path:
        sys.path.insert(0, emilia_path)
        
    from main import main_process, ModelPack
    from models import separate_fast, dnsmos, whisper_asr, silero_vad
    from utils.tool import load_cfg

    device_id_for_process = 0 
    
    s3_client = create_r2_client()
    print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Starting...")

    print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Loading all models into memory...")
    
    torch_device = torch.device(f"cuda:{device_id_for_process}")
    generic_cuda_device = "cuda"
    
    cfg = load_cfg(EMILIA_CONFIG_PATH)
    cfg["huggingface_token"] = os.getenv("HF_TOKEN")

    # Optional: reduce separation chunk size to lower peak memory
    if SEPARATION_CHUNKS_ENV:
        try:
            cfg["separate"]["step1"]["chunks"] = int(SEPARATION_CHUNKS_ENV)
            print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Using separation chunks={cfg['separate']['step1']['chunks']}")
        except Exception:
            pass

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
        device=(SEPARATION_DEVICE if SEPARATION_DEVICE in ("cuda", "cpu") else generic_cuda_device), 
        device_index=device_id_for_process
    )
    dnsmos_model = dnsmos.ComputeScore(
        cfg["mos_model"]["primary_model_path"], 
        device=(DNSMOS_DEVICE if DNSMOS_DEVICE in ("cuda", "cpu") else generic_cuda_device),
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
        episode_id = task['episode_id']
        language = task['language']
        audio_key = task['audio_key']
        print(f"Worker-{rank} on Physical-GPU-{assigned_gpu_id}: Claimed task for episode: {episode_id} ({language})")
        try:
            with tempfile.TemporaryDirectory(prefix=f"gpu_worker_{assigned_gpu_id}_{rank}_") as temp_dir:
                local_filename = os.path.basename(audio_key)
                local_audio_path = Path(temp_dir) / local_filename
                s3_client.download_file(S3_BUCKET, audio_key, str(local_audio_path))
                # Quick sanity check to avoid ffmpeg errors on empty/corrupt downloads
                try:
                    if local_audio_path.stat().st_size < 1024:
                        raise ValueError(f"Downloaded file too small: {local_audio_path} ({local_audio_path.stat().st_size} bytes)")
                except Exception as e:
                    raise
                emilia_output_dir = Path(temp_dir) / "processed"
                os.makedirs(emilia_output_dir, exist_ok=True)
                main_process(
                    audio_path=str(local_audio_path), models=models, cfg=cfg,
                    device=torch_device, batch_size=BATCH_SIZE,
                    save_path=str(emilia_output_dir), audio_name=episode_id
                )
                # Upload under processed/{language}/{episode_id}/
                safe_language = (language or '').strip() or 'unknown'
                s3_processed_prefix = f"{S3_PROCESSED_PREFIX}{safe_language}/{episode_id}"
                upload_directory_to_r2(s3_client, emilia_output_dir, S3_BUCKET, s3_processed_prefix)
            
            complete_processing_task(s3_client, task['key'])
            print(f"  Worker-{rank} on Physical-GPU-{assigned_gpu_id}: ✅ Finished and completed task for episode: {episode_id}")

        except Exception as e:
            print(f"  Worker-{rank} on Physical-GPU-{assigned_gpu_id}: [!!!] CRITICAL FAILURE on episode {episode_id}. Error: {e}")
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

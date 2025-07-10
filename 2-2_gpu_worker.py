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

# --- Configuration ---
S3_BUCKET = "yt-pipeline-bucket"
S3_RAW_AUDIO_PREFIX = "raw_audio/"
S3_PROCESSED_PREFIX = "processed/"
S3_TASKS_BASE_PREFIX = "tasks/"
MAX_EMILIA_WORKERS = 9999 # Number of GPU workers to run per instance
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"

def claim_processing_task(s3_client):
    """
    Atomically claims a task by listing and then attempting to delete it.
    The first worker to successfully delete the task from 'todo' wins the claim.
    """
    # This prefix points to the to-do list for the GPU workers
    todo_prefix = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_todo/')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=todo_prefix, MaxKeys=10)
    
    for page in pages:
        if 'Contents' not in page:
            continue
        
        tasks = page['Contents']
        random.shuffle(tasks)

        for obj in tasks:
            task_key = obj['Key']
            if not task_key.endswith('.task'):
                continue
            
            try:
                # First, get the content of the task file (the video_id)
                task_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=task_key)
                video_id = task_obj['Body'].read().decode('utf-8')
                
                # Now, attempt the atomic delete operation to claim the task.
                s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
                
                # If the delete succeeded, we have claimed the task.
                # We now create the 'in_progress' file to track it.
                in_progress_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_in_progress', f"{video_id}.task")
                s3_client.put_object(Bucket=S3_BUCKET, Key=in_progress_key, Body=video_id)

                return {'key': in_progress_key, 'video_id': video_id}

            except ClientError as e:
                # If we get a NoSuchKey error, it means another worker deleted it first.
                # We lost the race, so we just continue to the next available task.
                if e.response['Error']['Code'] in ['NoSuchKey', '404']:
                    continue
                else:
                    raise # unexpected errors
    return None

def complete_processing_task(s3_client, task_key):
    """
    Moves a completed processing task file to the 'processing_completed' directory.
    This is now robust against race conditions where another worker completes the task first.
    """
    video_id = os.path.basename(task_key).replace('.task', '')
    completed_key = os.path.join(S3_TASKS_BASE_PREFIX, 'processing_completed', f"{video_id}.task")
    try:
        # First, check if the in-progress file still exists before trying to move it.
        s3_client.head_object(Bucket=S3_BUCKET, Key=task_key)
        
        # If it exists, it means we are the first worker to finish. Move it.
        s3_client.copy_object(
            Bucket=S3_BUCKET,
            CopySource={'Bucket': S3_BUCKET, 'Key': task_key},
            Key=completed_key
        )
        s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)

    except ClientError as e:
        # If head_object returns a 404 error, it means another worker already completed this task.
        # This is an expected outcome of the race condition, so we can safely ignore it.
        if e.response['Error']['Code'] == '404':
            print(f"  Note: Task for {video_id} was already completed by another worker.")
        else:
            # If it's a different error, we should raise it to be aware of other potential issues.
            print(f"  Warning: Unexpected S3 error while completing task {task_key}: {e}")
            raise

def run_emilia_pipe(input_flac_file: str, output_dir: str, device: str):
    """Runs the Emilia-pipe on a specific audio file using a specific GPU."""
    os.makedirs(output_dir, exist_ok=True)
    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline" # Assumes this conda env exists
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={device} && \
    python {emilia_script} --input_file_path '{input_flac_file}' --config_path '{EMILIA_CONFIG_PATH}' --output_dir '{output_dir}' --quiet
    """
    try:
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Emilia error on GPU {device} for {input_flac_file}:\n---\n{e.stderr}\n---")
        raise

def processing_worker(rank: int, device: str):
    """A worker process that continuously claims and processes audio files."""
    s3_client = boto3.client('s3')
    print(f"GPU-{device} (Rank {rank}): Starting...")
    
    while True:
        task = claim_processing_task(s3_client)
        if not task:
            print(f"GPU-{device} (Rank {rank}): No more tasks found. Exiting.")
            break

        video_id = task['video_id']
        print(f"GPU-{device} (Rank {rank}): Claimed task for video: {video_id}")

        try:
            with tempfile.TemporaryDirectory(prefix=f"gpu_worker_{device}_") as temp_dir:
                # 1. Download the raw audio file from S3
                input_flac_key = f"{S3_RAW_AUDIO_PREFIX}{video_id}.flac"
                local_flac_path = Path(temp_dir) / f"{video_id}.flac"
                print(f"  GPU-{device}: Downloading {input_flac_key}...")
                s3_client.download_file(S3_BUCKET, input_flac_key, str(local_flac_path))

                # 2. Run the Emilia processing pipeline
                emilia_output_dir = Path(temp_dir) / "processed"
                print(f"  GPU-{device}: Starting Emilia pipe for {video_id}...")
                run_emilia_pipe(str(local_flac_path), str(emilia_output_dir), device)

                # 3. Upload the processed results
                s3_processed_prefix = f"{S3_PROCESSED_PREFIX}{video_id}"
                print(f"  GPU-{device}: Uploading processed results for {video_id}...")
                upload_directory_to_s3(emilia_output_dir, S3_BUCKET, s3_processed_prefix)
            
            # 4. Mark the task as complete
            complete_processing_task(s3_client, task['key'])
            print(f"  GPU-{device}: ✅ Finished and completed task for video: {video_id}")

        except Exception as e:
            print(f"  GPU-{device}: [!!!] CRITICAL FAILURE on video {video_id}. Error: {e}")
            # The failed task remains in 'in_progress' for the janitor to handle.
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
        return ["0"]

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
    multiprocessing.set_start_method("spawn", force=True)
    main()

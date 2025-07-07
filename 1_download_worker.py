import os
import subprocess
import multiprocessing
import tempfile
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from yt_dlp import YoutubeDL
import time
import random

# --- Configuration ---
S3_BUCKET = "yt-pipeline-bucket"
# The S3 "folder" where the final raw audio files will be stored.
S3_RAW_AUDIO_PREFIX = "raw_audio/"
# The base "folder" for all our task files.
S3_TASKS_BASE_PREFIX = "tasks/"
# The audio format and sample rate we will standardize to.
SAMPLE_RATE = 24000
# Number of downloader processes to run per instance.
NUM_WORKERS =  multiprocessing.cpu_count()
MAX_CONSECUTIVE_FAILURES = 20
NUM_COOKIES = 4

def claim_video_task(s3_client):
    """
    Finds a video task in 'videos_todo/', moves it to 'videos_in_progress/',
    and returns the task details.
    """
    todo_prefix = os.path.join(S3_TASKS_BASE_PREFIX, 'videos_todo/')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=todo_prefix, MaxKeys=10)
    
    for page in pages:
        if 'Contents' not in page:
            continue
        
        tasks = page['Contents']
        random.shuffle(tasks) # Randomize to reduce contention between workers

        for obj in tasks:
            task_key = obj['Key']
            if not task_key.endswith('.task'):
                continue

            video_id = os.path.basename(task_key).replace('.task', '')
            in_progress_key = os.path.join(S3_TASKS_BASE_PREFIX, 'videos_in_progress', f"{video_id}.task")

            try:
                s3_client.copy_object(
                    Bucket=S3_BUCKET,
                    CopySource={'Bucket': S3_BUCKET, 'Key': task_key},
                    Key=in_progress_key
                )
                s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
                
                task_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=in_progress_key)
                video_url = task_obj['Body'].read().decode('utf-8')
                
                return {'key': in_progress_key, 'url': video_url, 'video_id': video_id}
            
            except ClientError as e:
                if e.response['Error']['Code'] in ['NoSuchKey', '404']:
                    continue
                else:
                    raise
    return None

def complete_video_task(s3_client, task_key):
    """
    Moves a completed video task file to the 'videos_completed' directory.
    This is now robust against race conditions where another worker completes the task first.
    """
    video_id = os.path.basename(task_key).replace('.task', '')
    completed_key = os.path.join(S3_TASKS_BASE_PREFIX, 'videos_completed', f"{video_id}.task")
    try:
        # Check if the in-progress file still exists before trying to move it.
        s3_client.head_object(Bucket=S3_BUCKET, Key=task_key)
        
        # If it exists, move it.
        s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': task_key}, Key=completed_key)
        s3_client.delete_object(Bucket=S3_BUCKET, Key=task_key)
    except ClientError as e:
        # If the key is not found, it means another worker already completed it. This is not an error.
        if e.response['Error']['Code'] == '404':
            print(f"  Note: Task {video_id} was already completed by another worker.")
        else:
            # Re-raise other unexpected S3 errors.
            raise

def download_and_convert_to_flac(video_url: str, temp_dir: Path, cookie_num: int):
    """Downloads a single video and converts it to a standardized FLAC file."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(temp_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'ignoreerrors': True,
        'cookiefile': '/home/ec2-user/cookies{cookie_num}.txt'
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        if not info:
            raise ValueError("yt-dlp returned no info")
        
        video_id = info['id']
        input_path = ydl.prepare_filename(info)
        output_path = temp_dir / f"{video_id}.flac"

        ffmpeg_cmd = [
            'ffmpeg', '-i', str(input_path), '-vn', '-ar', str(SAMPLE_RATE),
            '-ac', '1', '-sample_fmt', 's16', str(output_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return output_path

def downloader_worker(rank: int, failure_counter):
    """A worker process that continuously claims and processes video download tasks."""
    s3_client = boto3.client('s3')
    print(f"Downloader-{rank}: Starting...")
    
    while True:
        video_task = claim_video_task(s3_client)
        if failure_counter.value >= MAX_CONSECUTIVE_FAILURES:
            print(f"Downloader-{rank}: Max failures reached ({MAX_CONSECUTIVE_FAILURES}). Exiting.")
            break

        if not video_task:
            print(f"Downloader-{rank}: No more video tasks found. Exiting.")
            break

        video_id = video_task['video_id']
        video_url = video_task['url']
        print(f"Downloader-{rank}: Claimed task for video: {video_id}")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                flac_path = download_and_convert_to_flac(video_url, Path(temp_dir), rank % NUM_COOKIES)
                
                s3_key = f"{S3_RAW_AUDIO_PREFIX}{video_id}.flac"
                print(f"  Downloader-{rank}: Uploading {video_id}.flac to S3...")
                s3_client.upload_file(str(flac_path), S3_BUCKET, s3_key)
            
            complete_video_task(s3_client, video_task['key'])
            print(f"  Downloader-{rank}: ✅ Finished and completed task for video: {video_id}")
            failure_counter.value = 0

        except Exception as e:
            print(f"  Downloader-{rank}: [!!!] CRITICAL FAILURE on video {video_id}. Error: {e}")
            # --- MODIFIED: Increment the shared failure counter ---
            failure_counter.value += 1
            time.sleep(10)

def main():
    """Orchestrates the pool of downloader worker processes."""
    # --- MODIFIED: Create a manager and a shared failure counter ---
    manager = multiprocessing.Manager()
    failure_counter = manager.Value('i', 0)
    
    processes = []
    print(f"🚀 Starting {NUM_WORKERS} downloader worker processes...")
    for i in range(NUM_WORKERS):
        # --- MODIFIED: Pass the failure counter to each worker ---
        p = multiprocessing.Process(target=downloader_worker, args=(i, failure_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    # --- MODIFIED: Print the final failure count ---
    print("\n✅ All downloader workers have finished.")
    print(f"    Total failures: {failure_counter.value}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
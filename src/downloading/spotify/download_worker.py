import boto3
import os
import requests
import subprocess
import time
import random
import multiprocessing
from botocore.exceptions import ClientError
from urllib.parse import urlparse

# --- Configuration ---
S3_BUCKET_NAME = 'sptfy-dataset'
S3_TODO_PREFIX = 'tasks/download_todo/'
S3_IN_PROGRESS_PREFIX = 'tasks/download_in_progress/'
S3_COMPLETED_PREFIX = 'tasks/download_completed/'
S3_FAILED_PREFIX = 'tasks/download_failed/'
S3_OUTPUT_PREFIX = 'raw-audio/'

LOCAL_TEMP_DIR = 'temp_processing'
HEADERS = {'User-Agent': 'PodcastDatasetCrawler-AudioResearch/1.0'}

# Since the task is now purely network I/O bound, we can handle more concurrent workers.
# Adjust based on instance type and network performance.
NUM_WORKERS = 8

MAX_CONSECUTIVE_FAILURES = 20
IDLE_CHECK_THRESHOLD = 5

# Boto3 clients are not safe to share across processes, so each worker will create its own.

def claim_task():
    """
    Lists tasks in the 'todo' folder and attempts to atomically move one
    to the 'in_progress' folder. Returns the new key or None on failure.
    """
    s3_client = boto3.client('s3')
    
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_TODO_PREFIX, MaxKeys=50)
    if 'Contents' not in response:
        return None

    tasks = response['Contents']
    random.shuffle(tasks)

    for task in tasks:
        source_key = task['Key']
        try:
            episode_id = os.path.basename(source_key).replace('.task', '')
            in_progress_key = f"{S3_IN_PROGRESS_PREFIX}{episode_id}.task"

            s3_client.copy_object(
                Bucket=S3_BUCKET_NAME,
                CopySource={'Bucket': S3_BUCKET_NAME, 'Key': source_key},
                Key=in_progress_key
            )
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=source_key)
            
            return in_progress_key
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                continue
            else:
                raise
    return None


def process_task(task_key):
    """
    The core work function for a single task.
    Downloads the raw audio file and uploads it to S3.
    Returns True on success, False on failure.
    """
    s3_client = boto3.client('s3')
    episode_id = os.path.basename(task_key).replace('.task', '')
    
    local_download_path = None # Define here for the finally block

    try:
        # 1. Get download URL and determine original file extension
        task_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=task_key)
        url = task_obj['Body'].read().decode('utf-8')
        
        # Get file extension from URL path (e.g., .mp3, .m4a)
        path = urlparse(url).path
        extension = os.path.splitext(path)[1]
        if not extension:
            extension = ".mp3" # Default to .mp3 if no extension found

        local_download_path = os.path.join(LOCAL_TEMP_DIR, f"{episode_id}{extension}")
        
        # 2. Download the original file
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_download_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 3. Upload the original file directly to S3
        final_s3_key = f"{S3_OUTPUT_PREFIX}{episode_id}{extension}"
        s3_client.upload_file(local_download_path, S3_BUCKET_NAME, final_s3_key)
        
        # 4. Defensive Finalization (Success)
        try:
            completed_key = f"{S3_COMPLETED_PREFIX}{episode_id}.task"
            s3_client.copy_object(Bucket=S3_BUCKET_NAME, CopySource={'Bucket': S3_BUCKET_NAME, 'Key': task_key}, Key=completed_key)
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=task_key)
            print(f"🎉 Downloaded: {episode_id}{extension}")
            return True # Indicate success
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"⚠️ Note: Task {episode_id} was already completed by another worker.")
                return True # Treat as success
            else:
                raise

    except Exception as e:
        print(f"❌ FAILED: {episode_id}. Error: {e}")
        # 5. Defensive Finalization (Failure)
        try:
            failed_key = f"{S3_FAILED_PREFIX}{episode_id}.task"
            s3_client.copy_object(Bucket=S3_BUCKET_NAME, CopySource={'Bucket': S3_BUCKET_NAME, 'Key': task_key}, Key=failed_key)
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=task_key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                 print(f"⚠️ Note: Failed task {episode_id} was already moved by another worker.")
            else:
                print(f"❌ Critical error: Could not move failed task {episode_id}. Error: {e}")
        return False # Indicate failure
    finally:
        # 6. Cleanup local download file
        if local_download_path and os.path.exists(local_download_path):
            os.remove(local_download_path)


def worker_job(rank, failure_counter, lock):
    """
    The main loop for a single worker PROCESS. It continuously tries to claim and
    process tasks until it can't find any or the circuit breaker trips.
    """
    print(f"--- Worker-{rank} started (PID: {os.getpid()}) ---")
    idle_checks = 0
    while idle_checks < IDLE_CHECK_THRESHOLD:
        if failure_counter.value >= MAX_CONSECUTIVE_FAILURES:
            print(f"--- Worker-{rank}: Circuit breaker tripped! Shutting down. ---")
            break

        task_key = claim_task()
        if task_key:
            idle_checks = 0 # Reset idle counter on finding work
            success = process_task(task_key)
            if success:
                # On success, reset the shared failure counter
                failure_counter.value = 0
            else:
                # On failure, use the shared lock to safely increment the counter
                with lock:
                    failure_counter.value += 1
        else:
            # If no tasks are found, increment the idle counter and wait
            idle_checks += 1
            print(f"   Worker-{rank} idle (check {idle_checks}/{IDLE_CHECK_THRESHOLD})... sleeping.")
            time.sleep(30 + random.uniform(0, 30)) # Sleep with jitter
    
    print(f"--- Worker-{rank} shutting down after {IDLE_CHECK_THRESHOLD} idle checks. ---")


def main():
    """
    Initializes and runs the multiprocessing pool.
    """
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    
    with multiprocessing.Manager() as manager:
        failure_counter = manager.Value('i', 0)
        lock = manager.Lock() 
        
        print(f"--- Main process started. Launching {NUM_WORKERS} worker processes. ---")
        
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            worker_args = [(i, failure_counter, lock) for i in range(NUM_WORKERS)]
            pool.starmap(worker_job, worker_args)

    print("🏁 All worker processes have completed. Main process exiting. Goodbye!")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()

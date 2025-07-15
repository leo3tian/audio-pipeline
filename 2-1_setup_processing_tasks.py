import os
import boto3
import tqdm
import concurrent.futures

# --- Configuration ---
S3_BUCKET = "sptfy-dataset"
S3_RAW_AUDIO_PREFIX = "raw-audio/"
S3_TASKS_PREFIX = "tasks/processing_todo/"
SUPPORTED_EXTENSIONS = ('.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma')
# Adjust the number of concurrent workers as needed
MAX_WORKERS = 50

# Create a single S3 client to be shared across threads
s3_client = boto3.client("s3")

def create_s3_task(s3_key):
    """
    Creates a single task file in S3 for a given audio file key.
    This function will be executed by each worker thread.
    """
    try:
        if s3_key.endswith(SUPPORTED_EXTENSIONS):
            base_name = os.path.basename(s3_key)
            video_id = os.path.splitext(base_name)[0]
            
            task_key = f"{S3_TASKS_PREFIX}{video_id}.task"
            s3_client.put_object(Bucket=S3_BUCKET, Key=task_key, Body=video_id)
            return True # Indicate success
    except Exception as e:
        print(f"Error creating task for {s3_key}: {e}")
        return False # Indicate failure
    return False

def main():
    """
    Scans S3 and concurrently creates processing tasks for each audio file.
    """
    print(f"Starting task setup for audio files in s3://{S3_BUCKET}/{S3_RAW_AUDIO_PREFIX}")
    
    # Use a paginator to handle potentially millions of files
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_AUDIO_PREFIX)

    tasks_created = 0
    # The ThreadPoolExecutor manages our pool of worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list to hold all the jobs we want to run
        future_to_key = {executor.submit(create_s3_task, obj['Key']): obj['Key'] for page in pages for obj in page.get('Contents', [])}
        
        # Use tqdm to create a progress bar for the completed tasks
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_key), total=len(future_to_key), desc="Creating S3 tasks"):
            if future.result():
                tasks_created += 1

    if tasks_created == 0:
        print("\n[!] No supported audio files found. No tasks were created.")
        return

    print(f"\n✅ GPU task setup complete.")
    print(f"   {tasks_created} tasks created in s3://{S3_BUCKET}/{S3_TASKS_PREFIX}")

if __name__ == "__main__":
    main()
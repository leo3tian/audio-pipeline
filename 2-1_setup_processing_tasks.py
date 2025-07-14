import os
import boto3
import tqdm

# --- Configuration ---
S3_BUCKET = "sptfy-dataset"
# The location of the raw audio files from Stage 1
S3_RAW_AUDIO_PREFIX = "raw_audio/"
# The S3 "folder" where the GPU processing tasks will be created
S3_TASKS_PREFIX = "tasks/processing_todo/"
# A tuple of supported audio file extensions
SUPPORTED_EXTENSIONS = ('.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma')

def main():
    """
    Scans the raw_audio/ directory in S3 and creates a processing task
    for each supported audio file found.
    """
    print(f"Starting task setup for audio files in s3://{S3_BUCKET}/{S3_RAW_AUDIO_PREFIX}")
    s3_client = boto3.client("s3")
    
    # Use a paginator to handle potentially millions of files
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_AUDIO_PREFIX)

    tasks_created = 0
    for page in tqdm.tqdm(pages, desc="Scanning S3 for audio files"):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            # Check if the file ends with any of the supported extensions
            if obj['Key'].endswith(SUPPORTED_EXTENSIONS):
                # Robustly get the filename without the extension
                base_name = os.path.basename(obj['Key'])
                video_id = os.path.splitext(base_name)[0]
                
                task_key = f"{S3_TASKS_PREFIX}{video_id}.task"
                
                # The content of the task is just the video_id
                s3_client.put_object(Bucket=S3_BUCKET, Key=task_key, Body=video_id)
                tasks_created += 1
    
    if tasks_created == 0:
        print("\n[!] No supported audio files found. No tasks were created.")
        return

    print(f"\n✅ GPU task setup complete.")
    print(f"   {tasks_created} tasks created in s3://{S3_BUCKET}/{S3_TASKS_PREFIX}")

if __name__ == "__main__":
    main()
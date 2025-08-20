import os
import json
import boto3
import tqdm
import concurrent.futures
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
S3_BUCKET = os.environ.get("R2_BUCKET")
if not S3_BUCKET:
    raise ValueError("FATAL: The environment variable 'R2_BUCKET' is not set.")
S3_RAW_AUDIO_PREFIX = os.environ.get("RAW_AUDIO_PREFIX", "raw_audio/")
S3_TASKS_PREFIX = os.environ.get("TASKS_TODO_PREFIX", "tasks/processing_todo/")
SUPPORTED_EXTENSIONS = ('.flac', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma')
# Adjust the number of concurrent workers as needed
MAX_WORKERS = int(os.environ.get("TASKS_MAX_WORKERS", "200"))


def _create_s3_client():
    """Create an S3-compatible client for Cloudflare R2."""
    endpoint_url = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not endpoint_url:
        raise ValueError("R2 endpoint is required. Set R2_ENDPOINT_URL or R2_ENDPOINT.")
    if not (access_key and secret_key):
        raise ValueError("R2 credentials are required. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.")

    base_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})

    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        region_name="auto",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=base_config,
    )

# Create a single S3 client to be shared across threads
s3_client = _create_s3_client()


def create_s3_task(s3_key):
    """
    Creates a single task file in R2 for a given audio file key.
    """
    try:
        if s3_key.endswith(SUPPORTED_EXTENSIONS):
            base_name = os.path.basename(s3_key)
            episode_id = os.path.splitext(base_name)[0]

            # Derive language as the segment after 'raw_audio' in the key
            segments = s3_key.strip('/').split('/')
            language = None
            if 'raw_audio' in segments:
                idx = segments.index('raw_audio')
                if idx + 1 < len(segments):
                    language = segments[idx + 1]
            if not language:
                return False

            task_key = f"{S3_TASKS_PREFIX}{episode_id}.task"
            body = json.dumps({
                "episode_id": episode_id,
                "language": language,
                "audio_key": s3_key,
            })
            s3_client.put_object(Bucket=S3_BUCKET, Key=task_key, Body=body)
            return True
    except Exception as e:
        print(f"Error creating task for {s3_key}: {e}")
        return False
    return False


def main():
    """
    Scans R2 and concurrently creates processing tasks for each audio file,
    using streaming pagination and bounded in-flight submissions to avoid
    high memory usage for very large keysets.
    """
    print(f"Starting task setup for audio files in r2://{S3_BUCKET}/{S3_RAW_AUDIO_PREFIX}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_AUDIO_PREFIX)

    tasks_created = 0
    submitted = 0
    max_inflight = max(1, MAX_WORKERS * 4)
    inflight = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         tqdm.tqdm(desc="Creating R2 tasks", unit="task", dynamic_ncols=True) as pbar:

        def drain_until(target_size: int):
            nonlocal tasks_created
            to_remove = []
            for future in concurrent.futures.as_completed(list(inflight)):
                try:
                    if future.result():
                        tasks_created += 1
                        pbar.update(1)
                except Exception as e:
                    # Log and continue; this individual key failed to create a task.
                    # Avoid crashing the entire job.
                    # You can add more detailed logging if needed.
                    pass
                to_remove.append(future)
                if len(inflight) - len(to_remove) <= target_size:
                    break
            for f in to_remove:
                inflight.discard(f)

        for page in pages:
            contents = page.get('Contents', [])
            if not contents:
                continue
            for obj in contents:
                key = obj.get('Key')
                if not key:
                    continue
                inflight.add(executor.submit(create_s3_task, key))
                submitted += 1
                if len(inflight) >= max_inflight:
                    drain_until(max_inflight // 2)

        # Drain all remaining
        drain_until(0)

    if tasks_created == 0:
        print("\n[!] No supported audio files found. No tasks were created.")
        return

    print(f"\n✅ GPU task setup complete.")
    print(f"   {tasks_created} tasks created in r2://{S3_BUCKET}/{S3_TASKS_PREFIX}")

if __name__ == "__main__":
    main()
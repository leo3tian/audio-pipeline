import json
import boto3
import os

# --- Configuration ---
S3_BUCKET_NAME = 'sptfy-dataset'  # 👈 The name of your S3 bucket
S3_KEY_FOR_JSONL = 'source-data/spotify_podcast_data.jsonl'  # 👈 The "path" to the jsonl file

# Define all prefixes where tasks could possibly exist
TASK_PREFIXES_TO_CHECK = [
    'tasks/download_todo/',
    'tasks/download_in_progress/',
    'tasks/download_completed/',
    'tasks/download_failed/'
]
S3_TODO_PREFIX = 'tasks/download_todo/'

# Initialize Boto3 S3 client
s3_client = boto3.client('s3')

def get_existing_task_ids(bucket, prefixes):
    """
    Efficiently lists all objects across multiple prefixes and returns a set of episode IDs.
    """
    existing_ids = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for prefix in prefixes:
        print(f"   Listing existing tasks in: {prefix}...")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                # Extract episode_id from a key like 'tasks/todo/episode123.task'
                episode_id = os.path.basename(obj['Key']).replace('.task', '')
                existing_ids.add(episode_id)
    return existing_ids

def main():
    """
    Reads the source dataset from S3 and creates task files for any new episodes.
    """
    print("--- Starting Task Generation ---")
    
    # 1. Get a set of all task IDs that have already been created
    print("STEP 1: Checking for all previously created tasks...")
    existing_ids = get_existing_task_ids(S3_BUCKET_NAME, TASK_PREFIXES_TO_CHECK)
    print(f"✅ Found {len(existing_ids):,} existing tasks across all stages.")

    # 2. Stream the source dataset and create tasks for new items
    print(f"\nSTEP 2: Reading source file and creating new tasks...")
    source_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_KEY_FOR_JSONL)
    
    tasks_created = 0
    lines_read = 0
    
    for line in source_obj['Body'].iter_lines():
        try:
            lines_read += 1
            episode = json.loads(line.decode('utf-8'))
            episode_id = episode.get('id')
            url = episode.get('enclosure_url')

            if not episode_id or not url:
                continue

            # This is the fast, in-memory check. No network call needed.
            if episode_id not in existing_ids:
                task_key = f"{S3_TODO_PREFIX}{episode_id}.task"
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=task_key, Body=url)
                tasks_created += 1
                # Add the new ID to our set so we don't duplicate it in this run
                existing_ids.add(episode_id) 

                if tasks_created > 0 and tasks_created % 1000 == 0:
                    print(f"   Created {tasks_created:,} new tasks...")

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"❌ Error decoding JSON on line {lines_read}: {e}")
            continue

    print("\n🎉 Task generation complete!")
    print(f"   Total lines read from source: {lines_read:,}")
    print(f"   New tasks created in this run: {tasks_created:,}")

if __name__ == "__main__":
    main()
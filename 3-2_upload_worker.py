import os
import json
import tqdm
from pathlib import Path
from huggingface_hub import HfApi
import boto3
from botocore.exceptions import ClientError
import tempfile
import tarfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/your-dataset-name")
HF_TOKEN = os.environ.get("HF_TOKEN")
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
if not SQS_QUEUE_URL:
    raise ValueError("FATAL: The environment variable 'SQS_QUEUE_URL' is not set.")

DOWNLOAD_WORKERS = 128
MAX_EMPTY_RECEIVES = 100 # Number of consecutive empty SQS receives before exiting

def download_file(s3_client, s3_key, local_path):
    """Helper function to download a single file for the thread pool."""
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        s3_client.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
             # Try without leading zeros
            path_parts = s3_key.rsplit('_', 1)
            if len(path_parts) == 2:
                base_path, segment_part = path_parts
                segment_num_str = segment_part.split('.')[0]
                if segment_num_str.startswith('0') and len(segment_num_str) > 1:
                    try:
                        alternative_key = f"{base_path}_{int(segment_num_str)}.mp3"
                        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=alternative_key)
                        s3_client.download_file(S3_BUCKET_NAME, alternative_key, str(local_path))
                        return True
                    except (ClientError, ValueError):
                        return False # Not found or parsing error
            return False
        raise

def create_and_upload_batch(api: HfApi, s3_client, batch_records: list, batch_num: int):
    """
    Downloads a batch of audio files in parallel, creates corresponding JSON files,
    archives them, and uploads the batch to Hugging Face.
    """
    batch_name = f"batch_{batch_num:05d}.tar"
    hf_tar_path = f"archives/{batch_name}"
    
    if api.file_exists(repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_tar_path):
        print(f"  Batch {batch_name} already exists on Hugging Face Hub. Skipping.")
        return True

    print(f"\nProcessing batch #{batch_num} with {len(batch_records)} files...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        batch_content_path = temp_path / "batch_content"
        os.makedirs(batch_content_path)

        valid_records = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            future_to_record = {
                executor.submit(
                    download_file, 
                    s3_client, 
                    record["_s3_key"], 
                    batch_content_path / os.path.basename(record["_s3_key"])
                ): record for record in batch_records
            }

            for future in tqdm.tqdm(as_completed(future_to_record), total=len(batch_records), desc=f"Downloading batch #{batch_num}"):
                record = future_to_record[future]
                try:
                    if future.result():
                        record_copy = record.copy()
                        record_copy.pop("_s3_key")
                        base_filename = record_copy['audio'].replace('.mp3', '')
                        json_filepath = batch_content_path / f"{base_filename}.json"
                        with open(json_filepath, 'w', encoding='utf-8') as f:
                            json.dump(record_copy, f, ensure_ascii=False, indent=2)
                        valid_records.append(record)
                    else:
                        print(f"  [!] Audio file not found: {record['_s3_key']}")
                except Exception as e:
                    print(f"  [!] Failed to process record {record.get('audio', 'N/A')}: {str(e)[:200]}")

        if not valid_records:
            print(f"  [!] No valid files in batch {batch_num}, skipping tar creation.")
            return True

        print(f"  Creating archive with {len(valid_records)} files...")
        tar_filepath = temp_path / batch_name
        with tarfile.open(tar_filepath, "w") as tar:
            tar.add(str(batch_content_path), arcname=".")
        
        print(f"  Uploading {batch_name} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=str(tar_filepath),
            path_in_repo=hf_tar_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"  ✅ Successfully uploaded {batch_name}")
    return True


def upload_worker():
    """
    A worker that continuously polls an SQS queue for upload tasks,
    processes them, and deletes them upon completion.
    """
    s3_client = boto3.client('s3')
    sqs_client = boto3.client('sqs')
    api = HfApi(token=HF_TOKEN)
    
    worker_id = ''.join(random.choices('0123456789ABCDEF', k=6))
    print(f"🚀 Worker-{worker_id}: Starting up...")
    
    empty_receives = 0
    while empty_receives < MAX_EMPTY_RECEIVES:
        print(f"\nWorker-{worker_id}: Polling for a new task...")
        response = sqs_client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20, # Use long polling
            VisibilityTimeout=3600 # 1 hour to process a large batch
        )

        if "Messages" not in response:
            empty_receives += 1
            print(f"  Worker-{worker_id}: Queue is empty. ({empty_receives}/{MAX_EMPTY_RECEIVES})")
            continue
        
        empty_receives = 0 # Reset counter on successful receive
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        batch_definition_key = message['Body']
        
        try:
            batch_num = int(batch_definition_key.split('_')[-1].split('.')[0])
            print(f"Worker-{worker_id}: Claimed task for batch #{batch_num} ({batch_definition_key})")

            # Download the batch definition file
            batch_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=batch_definition_key)
            batch_content = batch_obj['Body'].read().decode('utf-8')
            batch_records = [json.loads(line) for line in batch_content.strip().split('\n')]

            success = create_and_upload_batch(api, s3_client, batch_records, batch_num)

            if success:
                print(f"Worker-{worker_id}: Task for batch #{batch_num} completed successfully. Deleting message.")
                sqs_client.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            else:
                 print(f"Worker-{worker_id}: [!!!] Task for batch #{batch_num} failed. Message will become visible again.")

        except Exception as e:
            print(f"Worker-{worker_id}: [!!!] CRITICAL FAILURE on task {batch_definition_key}. Error: {e}")
            # Let the visibility timeout expire so another worker can try.
            time.sleep(60)

    print(f"\n✅ Worker-{worker_id}: No new tasks for {MAX_EMPTY_RECEIVES * 20} seconds. Shutting down.")


if __name__ == "__main__":
    upload_worker()

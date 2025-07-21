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
import itertools
from typing import Iterator, List, Dict
import queue
import threading

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PROCESSED_PREFIX = "processed/"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/your-dataset-name")
HF_TOKEN = os.environ.get("HF_TOKEN")
FILES_PER_TAR_BATCH = 5000
DOWNLOAD_WORKERS = 32
METADATA_SCAN_WORKERS = 64
PREFIXES_PER_BATCH = 1000  # Process S3 listing in batches

def download_file(s3_client, s3_key, local_path):
    """Helper function to download a single file for the thread pool."""
    s3_client.download_file(S3_BUCKET_NAME, s3_key, str(local_path))

def create_and_upload_batch(api: HfApi, s3_client, batch_records: list, batch_num: int):
    """
    Downloads a batch of audio files in parallel, creates corresponding JSON files,
    archives them, and uploads the batch to Hugging Face.
    """
    batch_name = f"batch_{batch_num:05d}.tar"
    hf_tar_path = f"archives/{batch_name}"
    
    if api.file_exists(repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_tar_path):
        print(f"  Batch {batch_name} already exists on Hugging Face Hub. Skipping.")
        return

    print(f"\nProcessing batch #{batch_num} with {len(batch_records)} files...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        batch_content_path = temp_path / "batch_content"
        os.makedirs(batch_content_path)

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            future_to_record = {
                executor.submit(
                    download_file, 
                    s3_client, 
                    record["_s3_key"], 
                    batch_content_path / os.path.basename(record["_s3_key"])
                ): record
                for record in batch_records
            }

            for future in tqdm.tqdm(as_completed(future_to_record), total=len(batch_records), desc=f"Downloading batch #{batch_num}"):
                record = future_to_record[future]
                try:
                    future.result()
                    
                    record.pop("_s3_key")
                    base_filename = record['audio'].replace('.mp3', '')
                    json_filepath = batch_content_path / f"{base_filename}.json"
                    with open(json_filepath, 'w', encoding='utf-8', errors='replace') as f:
                        json.dump(record, f, ensure_ascii=False, indent=2, errors='replace')

                except Exception as e:
                    print(f"  [!] Failed to download or process file for record {record.get('id', 'N/A')}. Error: {e}")

        tar_filepath = temp_path / batch_name
        print(f"  Creating mixed-content archive: {tar_filepath}")
        with tarfile.open(tar_filepath, "w") as tar:
            tar.add(str(batch_content_path), arcname=".")
        
        print(f"  Uploading {batch_name} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=str(tar_filepath),
            path_in_repo=hf_tar_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"  ✅ Successfully uploaded {batch_name}.")

def list_prefixes_generator(s3_client) -> Iterator[str]:
    """Generator that yields prefixes in batches to avoid memory issues."""
    paginator = s3_client.get_paginator('list_objects_v2')
    kwargs = {
        'Bucket': S3_BUCKET_NAME,
        'Prefix': S3_PROCESSED_PREFIX,
        'Delimiter': '/'
    }
    
    for page in paginator.paginate(**kwargs):
        for prefix in page.get('CommonPrefixes', []):
            if prefix:
                yield prefix['Prefix']

def process_metadata_batch(metadata_queue: queue.Queue, output_file, counter: Dict, stop_event: threading.Event):
    """Process metadata files from the queue and write to output file."""
    s3_client = boto3.client('s3')
    while not stop_event.is_set():
        try:
            prefix = metadata_queue.get(timeout=1.0)
            try:
                metadata_s3_key = os.path.join(prefix, "all_segments.json")
                response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
                # Explicitly decode the S3 response as UTF-8
                content = response['Body'].read().decode('utf-8', errors='replace')
                segments = json.loads(content)
                
                video_id = prefix.rstrip('/').split('/')[-1]
                for i, segment in enumerate(segments):
                    base_filename = f"{video_id}_{i:06d}"
                    s3_audio_key = os.path.join(prefix, f"{base_filename}.mp3")
                    
                    record = {
                        "audio": f"{base_filename}.mp3",
                        "text": segment.get("text", ""),
                        "speaker_id": segment.get("speaker", "UNKNOWN"),
                        "duration": segment.get("end", 0) - segment.get("start", 0),
                        "dnsmos": segment.get("dnsmos", 0.0),
                        "language": segment.get("language", "UNKNOWN"),
                        "_s3_key": s3_audio_key
                    }
                    
                    # Remove errors parameter from json.dumps
                    line = json.dumps(record, ensure_ascii=False) + '\n'
                    output_file.write(line)
                    with counter['lock']:
                        counter['total'] += 1
                        
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"  [!] S3 error fetching {prefix}: {e}")
            except json.JSONDecodeError as e:
                print(f"  [!] JSON decode error in {prefix}: {e}")
            except Exception as e:
                print(f"  [!] Error processing {prefix}: {e}")
            finally:
                metadata_queue.task_done()
        except queue.Empty:
            continue

def finalize_and_upload():
    """Optimized version for handling massive datasets."""
    s3_client = boto3.client('s3')
    api = HfApi(token=HF_TOKEN)
    
    print(f"🚀 Ensuring repository '{HF_REPO_ID}' exists...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_metadata_path = Path(temp_dir) / "master_metadata.jsonl"
        counter = {'total': 0, 'lock': threading.Lock()}
        stop_event = threading.Event()
        
        print("📝 Processing metadata (with streaming)...")
        metadata_queue = queue.Queue(maxsize=PREFIXES_PER_BATCH * 2)
        
        # Start the metadata processing workers
        with open(master_metadata_path, 'w', encoding='utf-8', errors='replace') as f_meta:
            with ThreadPoolExecutor(max_workers=METADATA_SCAN_WORKERS) as executor:
                workers = [
                    executor.submit(process_metadata_batch, metadata_queue, f_meta, counter, stop_event)
                    for _ in range(METADATA_SCAN_WORKERS)
                ]
                
                try:
                    prefix_count = 0
                    for prefix in list_prefixes_generator(s3_client):
                        metadata_queue.put(prefix)
                        prefix_count += 1
                        if prefix_count % 1000 == 0:
                            print(f"  Listed {prefix_count} prefixes...")
                    
                    metadata_queue.join()
                finally:
                    stop_event.set()
                    for worker in workers:
                        worker.result()

        total_segments = counter['total']
        if total_segments == 0:
            print("[!] No valid segments found.")
            return

        print(f"\nProcessing {total_segments} total segments...")
        num_batches = math.ceil(total_segments / FILES_PER_TAR_BATCH)
        
        # Process and upload in batches
        with open(master_metadata_path, 'r', encoding='utf-8', errors='replace') as f_meta:
            for i in range(num_batches):
                batch_lines = list(itertools.islice(f_meta, FILES_PER_TAR_BATCH))
                if not batch_lines: break
                
                batch_records = []
                for line in batch_lines:
                    try:
                        record = json.loads(line.strip())
                        batch_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"  [!] Skipping malformed JSON line: {e}")
                        continue
                
                if batch_records:  # Only process if we have valid records
                    create_and_upload_batch(api, s3_client, batch_records, i)
        
        print("\n✨ Upload complete!")
        print(f"Check your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    finalize_and_upload()

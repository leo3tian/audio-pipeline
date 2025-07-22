import os
import json
import tqdm
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import tempfile
import math
from concurrent.futures import ThreadPoolExecutor
import itertools
from typing import Iterator, Dict
import queue
import threading

# --- Configuration ---
# export S3_BUCKET_NAME=sptfy-dataset && export SQS_QUEUE_URL="https://sqs.us-east-2.amazonaws.com/450282239172/huggingface_upload"
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PROCESSED_PREFIX = "processed/"
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
if not SQS_QUEUE_URL:
    raise ValueError("FATAL: The environment variable 'SQS_QUEUE_URL' is not set.")
AWS_REGION = os.environ.get("AWS_REGION")
if not AWS_REGION:
    raise ValueError("FATAL: The environment variable 'AWS_REGION' is not set.")
S3_UPLOAD_TASKS_PREFIX = "tasks/upload_batches/" 

FILES_PER_TAR_BATCH = 20000
METADATA_SCAN_WORKERS = 256
PREFIXES_PER_BATCH = 2000

def list_prefixes_generator(s3_client) -> Iterator[str]:
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
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION"))
    while not stop_event.is_set():
        try:
            prefix = metadata_queue.get(timeout=1.0)
            try:
                metadata_s3_key = os.path.join(prefix, "all_segments.json")
                response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
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
                    
                    try:
                        json_str = json.dumps(record, ensure_ascii=False)
                        json.loads(json_str)
                        output_file.write(json_str + '\n')
                        output_file.flush()
                        with counter['lock']:
                            counter['total'] += 1
                    except Exception as e:
                        print(f"  [!] Failed to serialize record for {base_filename}: {e}")
                        print(f"  [DEBUG] Problem record: {str(record)[:200]}")
                        continue
                        
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

def setup_upload_tasks():
    """
    Scans S3 for all processed audio, creates batched task definition files,
    uploads them to S3, and sends a message to SQS for each task.
    This script should only be run ONCE to set up the entire upload job.
    """
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_metadata_path = Path(temp_dir) / "master_metadata.jsonl"
        counter = {'total': 0, 'lock': threading.Lock()}
        stop_event = threading.Event()
        
        print("PHASE 1: Scanning all metadata from S3...")
        metadata_queue = queue.Queue(maxsize=PREFIXES_PER_BATCH * 2)
        
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
            print("\n[!] No valid segments found. No tasks were created.")
            return

        print(f"\nPHASE 2: Creating and uploading task definitions...")
        num_batches = math.ceil(total_segments / FILES_PER_TAR_BATCH)
        print(f"  Total segments: {total_segments}")
        print(f"  Creating {num_batches} tasks of up to {FILES_PER_TAR_BATCH} files each.")

        sqs_messages = []
        with open(master_metadata_path, 'r', encoding='utf-8', errors='replace') as f_meta:
            for i in tqdm.tqdm(range(num_batches), desc="Creating and Uploading Tasks"):
                batch_lines = list(itertools.islice(f_meta, FILES_PER_TAR_BATCH))
                if not batch_lines:
                    break
                
                batch_definition_content = "".join(batch_lines)
                batch_definition_key = f"{S3_UPLOAD_TASKS_PREFIX}batch_{i:05d}.jsonl"

                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=batch_definition_key,
                    Body=batch_definition_content.encode('utf-8')
                )

                sqs_messages.append({
                    'Id': f'batch_{i}',
                    'MessageBody': batch_definition_key
                })

                if len(sqs_messages) == 10:
                    sqs_client.send_message_batch(QueueUrl=SQS_QUEUE_URL, Entries=sqs_messages)
                    sqs_messages = []

        if sqs_messages:
            sqs_client.send_message_batch(QueueUrl=SQS_QUEUE_URL, Entries=sqs_messages)

        print(f"\n✅ Task setup complete.")
        print(f"  {num_batches} upload tasks have been sent to the SQS queue.")
        print(f"  You can now launch your '3-2_upload_worker.py' instances.")

if __name__ == "__main__":
    setup_upload_tasks()

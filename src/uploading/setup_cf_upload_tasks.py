import os
import json
import tqdm
from pathlib import Path
import tempfile
import math
import itertools
from typing import Iterator, Dict
import queue
import threading
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import json as _json
import urllib.request
import urllib.error
from dotenv import load_dotenv
load_dotenv()


# --- Configuration ---
# Source: Cloudflare R2 (S3-compatible)
R2_BUCKET = os.environ.get("R2_BUCKET")
if not R2_BUCKET:
    raise ValueError("FATAL: The environment variable 'R2_BUCKET' is not set.")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT")
if not R2_ENDPOINT_URL:
    raise ValueError("FATAL: R2 endpoint is required. Set R2_ENDPOINT_URL or R2_ENDPOINT.")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
    raise ValueError("FATAL: R2 credentials are required. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.")

# Layout
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
# Where to write batch definitions in R2
UPLOAD_TASKS_PREFIX = os.environ.get("CF_UPLOAD_TASKS_PREFIX", "tasks/cf_upload_batches/todo/")

# Batching
FILES_PER_TAR_BATCH = int(os.environ.get("FILES_PER_TAR_BATCH", "20000"))
METADATA_SCAN_WORKERS = int(os.environ.get("METADATA_SCAN_WORKERS", "256"))
PREFIXES_PER_BATCH = int(os.environ.get("PREFIXES_PER_BATCH", "2000"))

# Optional: Cloudflare Queues publishing
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN")
CF_QUEUE_NAME = os.environ.get("CF_QUEUE_NAME")  # The target queue that Workers consume


def _create_r2_client():
    """Create an S3-compatible client for Cloudflare R2."""
    base_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    return boto3.client(
        service_name="s3",
        endpoint_url=R2_ENDPOINT_URL,
        region_name="auto",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=base_config,
    )


def list_language_prefixes(s3_client) -> Iterator[str]:
    """Yield per-episode prefixes under processed/<language>/ using Delimiter to avoid deep listings."""
    paginator = s3_client.get_paginator('list_objects_v2')
    # First list languages
    lang_kwargs = {
        'Bucket': R2_BUCKET,
        'Prefix': PROCESSED_PREFIX,
        'Delimiter': '/',
    }
    for lang_page in paginator.paginate(**lang_kwargs):
        for lang_prefix in lang_page.get('CommonPrefixes', []):
            language_root = lang_prefix['Prefix']
            # Now list episodes within this language
            ep_kwargs = {
                'Bucket': R2_BUCKET,
                'Prefix': language_root,
                'Delimiter': '/',
            }
            for ep_page in paginator.paginate(**ep_kwargs):
                for ep_prefix in ep_page.get('CommonPrefixes', []):
                    yield ep_prefix['Prefix']


def process_metadata_batch(s3_client, metadata_queue: queue.Queue, output_file, counter: Dict, stop_event: threading.Event):
    """Read all_segments.json for each episode prefix and append JSONL records to output_file."""
    while not stop_event.is_set():
        try:
            prefix = metadata_queue.get(timeout=1.0)
            try:
                metadata_s3_key = os.path.join(prefix, "all_segments.json")
                response = s3_client.get_object(Bucket=R2_BUCKET, Key=metadata_s3_key)
                content = response['Body'].read().decode('utf-8', errors='replace')
                segments = json.loads(content)

                episode_id = prefix.rstrip('/').split('/')[-1]
                for i, segment in enumerate(segments):
                    base_filename = f"{episode_id}_{i:06d}"
                    r2_audio_key = os.path.join(prefix, f"{base_filename}.mp3")

                    record = {
                        "audio": f"{base_filename}.mp3",
                        "text": segment.get("text", ""),
                        "speaker_id": segment.get("speaker", "UNKNOWN"),
                        "duration": segment.get("end", 0) - segment.get("start", 0),
                        "dnsmos": segment.get("dnsmos", 0.0),
                        "language": segment.get("language", "UNKNOWN"),
                        "_r2_key": r2_audio_key,
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
                        continue
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"  [!] R2 error fetching {prefix}: {e}")
            except json.JSONDecodeError as e:
                print(f"  [!] JSON decode error in {prefix}: {e}")
            except Exception as e:
                print(f"  [!] Error processing {prefix}: {e}")
            finally:
                metadata_queue.task_done()
        except queue.Empty:
            continue


def main():
    """
    Scan Cloudflare R2 for processed outputs and write Cloudflare-Worker-ready
    batch definition files to R2 under tasks/cf_upload_batches/todo/.
    Each batch is a JSONL listing of records with `_r2_key` pointing to the audio.
    """
    s3_client = _create_r2_client()

    with tempfile.TemporaryDirectory() as temp_dir:
        master_metadata_path = Path(temp_dir) / "master_metadata.jsonl"
        counter = {'total': 0, 'lock': threading.Lock()}
        stop_event = threading.Event()

        print("PHASE 1: Scanning all metadata from R2...")
        metadata_queue = queue.Queue(maxsize=PREFIXES_PER_BATCH * 2)

        with open(master_metadata_path, 'w', encoding='utf-8', errors='replace') as f_meta:
            workers = []
            for _ in range(METADATA_SCAN_WORKERS):
                t = threading.Thread(target=process_metadata_batch, args=(s3_client, metadata_queue, f_meta, counter, stop_event), daemon=True)
                t.start()
                workers.append(t)

            try:
                prefix_count = 0
                for prefix in list_language_prefixes(s3_client):
                    metadata_queue.put(prefix)
                    prefix_count += 1
                    if prefix_count % 1000 == 0:
                        print(f"  Listed {prefix_count} episode prefixes...")

                metadata_queue.join()
            finally:
                stop_event.set()
                for t in workers:
                    t.join()

        total_segments = counter['total']
        if total_segments == 0:
            print("\n[!] No valid segments found. No tasks were created.")
            return

        print(f"\nPHASE 2: Creating and uploading batch definitions to R2...")
        num_batches = math.ceil(total_segments / FILES_PER_TAR_BATCH)
        print(f"  Total segments: {total_segments}")
        print(f"  Creating {num_batches} tasks of up to {FILES_PER_TAR_BATCH} files each.")

        messages_to_send = []
        with open(master_metadata_path, 'r', encoding='utf-8', errors='replace') as f_meta:
            for i in tqdm.tqdm(range(num_batches), desc="Uploading R2 batch definitions"):
                batch_lines = list(itertools.islice(f_meta, FILES_PER_TAR_BATCH))
                if not batch_lines:
                    break

                batch_definition_content = "".join(batch_lines)
                batch_key = f"{UPLOAD_TASKS_PREFIX}batch_{i:05d}.jsonl"

                s3_client.put_object(
                    Bucket=R2_BUCKET,
                    Key=batch_key,
                    Body=batch_definition_content.encode('utf-8')
                )

                # Optionally enqueue a Cloudflare Queue message for this batch
                if CF_ACCOUNT_ID and CF_API_TOKEN and CF_QUEUE_NAME:
                    messages_to_send.append({"body": batch_key})
                    # CF Queues API max 100 messages; flush in chunks to reduce memory
                    if len(messages_to_send) >= 100:
                        _publish_cf_queue_messages(messages_to_send)
                        messages_to_send.clear()

        print(f"\n✅ Task setup complete.")
        print(f"  {num_batches} R2 batch definitions created under r2://{R2_BUCKET}/{UPLOAD_TASKS_PREFIX}")
        if CF_ACCOUNT_ID and CF_API_TOKEN and CF_QUEUE_NAME:
            if messages_to_send:
                _publish_cf_queue_messages(messages_to_send)
                messages_to_send.clear()
            print(f"  Enqueued {num_batches} messages to Cloudflare Queue '{CF_QUEUE_NAME}'.")
        else:
            print("  Note: CF_ACCOUNT_ID/CF_API_TOKEN/CF_QUEUE_NAME not set; no queue messages were published.")


if __name__ == "__main__":
    main()


def _publish_cf_queue_messages(messages):
    """Publish messages to Cloudflare Queue via REST API."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/queues/{CF_QUEUE_NAME}/messages"
    payload = {"messages": messages}
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {CF_API_TOKEN}")
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 200:
                raise RuntimeError(f"CF Queue publish failed: HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"CF Queue publish HTTP error: {e.code} {e.read().decode('utf-8', 'ignore')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"CF Queue publish URL error: {e.reason}")



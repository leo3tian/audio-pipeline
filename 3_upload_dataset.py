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
import itertools # Import the itertools module

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PROCESSED_PREFIX = "processed/"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/your-dataset-name")
HF_TOKEN = os.environ.get("HF_TOKEN")
FILES_PER_TAR_BATCH = 5000
DOWNLOAD_WORKERS = 32

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
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)

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

def finalize_and_upload():
    """
    Scans S3, streams metadata to a local file to handle large datasets, 
    and then creates/uploads batched .tar archives to Hugging Face.
    """
    s3_client = boto3.client('s3')
    api = HfApi(token=HF_TOKEN)
    
    print(f"🚀 Ensuring repository '{HF_REPO_ID}' exists on Hugging Face Hub...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        master_metadata_path = Path(temp_dir) / "master_metadata.jsonl"
        total_segments = 0

        print("📝 Aggregating all metadata from S3 to local disk...")
        with open(master_metadata_path, 'w', encoding='utf-8') as f_meta:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PROCESSED_PREFIX, Delimiter='/')
            video_prefixes = [p['Prefix'] for p in pages.search('CommonPrefixes') if p]
            
            for s3_video_prefix in tqdm.tqdm(video_prefixes, desc="Scanning metadata"):
                video_id = s3_video_prefix.rstrip('/').split('/')[-1]
                metadata_s3_key = os.path.join(s3_video_prefix, "all_segments.json")
                try:
                    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
                    segments = json.loads(response['Body'].read().decode('utf-8'))
                    for i, segment in enumerate(segments):
                        base_filename = f"{video_id}_{i:06d}"
                        s3_audio_key = os.path.join(s3_video_prefix, f"{base_filename}.mp3")
                        
                        record = {
                            "audio": f"{base_filename}.mp3", "text": segment.get("text", ""),"speaker_id": segment.get("speaker", "UNKNOWN"), "duration": segment.get("end", 0) - segment.get("start", 0),"dnsmos": segment.get("dnsmos", 0.0), "language": segment.get("language", "UNKNOWN"),
                            "_s3_key": s3_audio_key
                        }
                        f_meta.write(json.dumps(record, ensure_ascii=False) + '\n')
                        total_segments += 1
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey': continue
                    else: raise

        if total_segments == 0:
            print("[!] No valid segments found to create a dataset.")
            return

        num_batches = math.ceil(total_segments / FILES_PER_TAR_BATCH)
        print(f"\nTotal segments to process: {total_segments}")
        print(f"Creating {num_batches} tar batches of up to {FILES_PER_TAR_BATCH} files each.")

        # --- SCALABILITY FIX: Read the master file in chunks using itertools ---
        with open(master_metadata_path, 'r', encoding='utf-8') as f_meta:
            for i in range(num_batches):
                # Use islice to safely grab a batch of lines from the file iterator
                batch_lines = list(itertools.islice(f_meta, FILES_PER_TAR_BATCH))
                if not batch_lines:
                    break # Stop if there are no more lines to read
                
                batch_records = [json.loads(line) for line in batch_lines]
                create_and_upload_batch(api, s3_client, batch_records, i)
        
        print("\n🎉🎉🎉 Dataset finalization and upload complete! 🎉🎉🎉")
        print(f"Check your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    finalize_and_upload()

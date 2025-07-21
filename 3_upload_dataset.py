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

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PROCESSED_PREFIX = "processed/"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/your-dataset-name")
HF_TOKEN = os.environ.get("HF_TOKEN")
# How many audio/json pairs go into each .tar archive
FILES_PER_TAR_BATCH = 5000
# Number of parallel download threads
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
    
    # Resumability: Check if this batch already exists on the Hub
    if api.file_exists(repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_tar_path):
        print(f"  Batch {batch_name} already exists on Hugging Face Hub. Skipping.")
        return

    print(f"\nProcessing batch #{batch_num} with {len(batch_records)} files...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        batch_content_path = temp_path / "batch_content"
        os.makedirs(batch_content_path)

        # Step 1: Download audio files in parallel and create individual JSON files
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
                    future.result() # Check for download errors
                    
                    # Pop the temporary S3 key and create the JSON file
                    record.pop("_s3_key")
                    base_filename = record['audio'].replace('.mp3', '') # audio is now just 'video_id_000001.mp3'
                    json_filepath = batch_content_path / f"{base_filename}.json"
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"  [!] Failed to download or process file for record {record.get('id', 'N/A')}. Error: {e}")

        # Step 2: Create a single .tar archive containing both file types
        tar_filepath = temp_path / batch_name
        print(f"  Creating mixed-content archive: {tar_filepath}")
        with tarfile.open(tar_filepath, "w") as tar:
            tar.add(str(batch_content_path), arcname=".")
        
        # Step 3: Upload the single .tar file to Hugging Face
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
    Scans S3, aggregates metadata, and creates/uploads batched .tar archives
    containing both audio and individual JSON files to Hugging Face.
    """
    s3_client = boto3.client('s3')
    api = HfApi(token=HF_TOKEN)
    
    print(f"🚀 Ensuring repository '{HF_REPO_ID}' exists on Hugging Face Hub...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    print(f"🔍 Scanning s3://{S3_BUCKET_NAME}/{S3_PROCESSED_PREFIX} for processed videos...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PROCESSED_PREFIX, Delimiter='/')
    video_prefixes = [p['Prefix'] for p in pages.search('CommonPrefixes') if p]
    
    all_metadata = []
    print("📝 Aggregating all metadata from S3...")
    for s3_video_prefix in tqdm.tqdm(video_prefixes, desc="Scanning metadata"):
        video_id = s3_video_prefix.rstrip('/').split('/')[-1]
        metadata_s3_key = os.path.join(s3_video_prefix, "all_segments.json")
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
            segments = json.loads(response['Body'].read().decode('utf-8'))
            for i, segment in enumerate(segments):
                base_filename = f"{video_id}_{i:06d}"
                s3_audio_key = os.path.join(s3_video_prefix, f"{base_filename}.mp3")
                
                # This record will be written to the individual JSON file
                all_metadata.append({
                    "audio": f"{base_filename}.mp3", 
                    "text": segment.get("text", ""),
                    "speaker_id": segment.get("speaker", "UNKNOWN"), 
                    "duration": segment.get("end", 0) - segment.get("start", 0),
                    "dnsmos": segment.get("dnsmos", 0.0), 
                    "language": segment.get("language", "UNKNOWN"),
                    "_s3_key": s3_audio_key # Temporary key for downloading
                })
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey': continue
            else: raise

    if not all_metadata:
        print("[!] No valid segments found to create a dataset.")
        return

    num_batches = math.ceil(len(all_metadata) / FILES_PER_TAR_BATCH)
    print(f"\nTotal segments to process: {len(all_metadata)}")
    print(f"Creating {num_batches} tar batches of up to {FILES_PER_TAR_BATCH} files each.")

    for i in range(num_batches):
        start_index = i * FILES_PER_TAR_BATCH
        end_index = start_index + FILES_PER_TAR_BATCH
        batch_records = all_metadata[start_index:end_index]
        create_and_upload_batch(api, s3_client, batch_records, i)
        
    print("\n🎉🎉🎉 Dataset finalization and upload complete! 🎉🎉🎉")
    print(f"Check your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    finalize_and_upload()

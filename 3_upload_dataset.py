import os
import json
import tqdm
from pathlib import Path
from huggingface_hub import HfApi, HfFolder
import boto3
from botocore.exceptions import ClientError
import tempfile
import tarfile
import math
import random

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PROCESSED_PREFIX = "processed/"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "fixie-ai/joe_rogan_youtube")
HF_TOKEN = os.environ.get("HF_TOKEN")
FILES_PER_TAR_BATCH = 5000 
S3_TASKS_BASE_PREFIX = "tasks/"

def create_and_upload_tar_batch(api: HfApi, s3_client, batch_records: list, batch_num: int):
    """Downloads a batch of files from S3, creates a .tar archive, and uploads it to HF."""
    batch_name = f"batch_{batch_num:05d}.tar"
    hf_tar_path = f"archives/{batch_name}"
    
    # First, check if this batch already exists on the Hub
    if api.file_exists(repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_tar_path):
        print(f"  Batch {batch_name} already exists on Hugging Face Hub. Skipping.")
        return

    print(f"\nProcessing {batch_name} with {len(batch_records)} files...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_path = temp_path / "audio"
        os.makedirs(download_path)

        for record in tqdm.tqdm(batch_records, desc=f"Downloading for {batch_name}"):
            s3_key = record["_s3_key"]
            local_filename = os.path.basename(s3_key)
            s3_client.download_file(S3_BUCKET_NAME, s3_key, str(download_path / local_filename))

        tar_filepath = temp_path / batch_name
        print(f"  Creating archive: {tar_filepath}")
        with tarfile.open(tar_filepath, "w") as tar:
            tar.add(str(download_path), arcname="audio")
        
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
    Scans S3, aggregates metadata, and processes batches in a resumable manner
    to create the final Hugging Face dataset.
    """
    if not HF_TOKEN:
        HfFolder.save_token(os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    
    api = HfApi()
    s3_client = boto3.client('s3')

    print(f"🚀 Ensuring repository '{HF_REPO_ID}' exists on Hugging Face Hub...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    # --- 1. Aggregate All Metadata from S3 First ---
    print("📝 Aggregating all metadata from S3...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PROCESSED_PREFIX, Delimiter='/')
    video_prefixes = [p['Prefix'] for p in pages.search('CommonPrefixes') if p]
    
    all_metadata = []
    for s3_video_prefix in tqdm.tqdm(video_prefixes, desc="Scanning metadata"):
        video_id = s3_video_prefix.rstrip('/').split('/')[-1]
        metadata_s3_key = os.path.join(s3_video_prefix, "all_segments.json")
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
            segments = json.loads(response['Body'].read().decode('utf-8'))
            for i, segment in enumerate(segments):
                mp3_filename = f"{video_id}_{i:06d}.mp3"
                hf_audio_path = f"audio/{mp3_filename}"
                s3_audio_key = os.path.join(s3_video_prefix, mp3_filename)
                all_metadata.append({
                    "id": f"{video_id}_{i:06d}", "audio": hf_audio_path, "text": segment.get("text", ""),
                    "speaker_id": segment.get("speaker", "UNKNOWN"), "duration": segment.get("end", 0) - segment.get("start", 0),
                    "dnsmos": segment.get("dnsmos", 0.0), "language": segment.get("language", "UNKNOWN"),
                    "_s3_key": s3_audio_key
                })
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey': continue
            else: raise

    if not all_metadata:
        print("[!] No valid segments found to create a dataset.")
        return

    # --- 2. Process Audio Batches Resumably ---
    num_batches = math.ceil(len(all_metadata) / FILES_PER_TAR_BATCH)
    print(f"\nTotal audio files to process: {len(all_metadata)}")
    print(f"This will be split into {num_batches} .tar batches.")

    for i in range(num_batches):
        start_index = i * FILES_PER_TAR_BATCH
        end_index = start_index + FILES_PER_TAR_BATCH
        batch_records = all_metadata[start_index:end_index]
        # In this simplified resumable version, we just check if the output file exists.
        # A full S3-task-based system could be added here if needed.
        create_and_upload_tar_batch(api, s3_client, batch_records, i)

    # --- 3. Upload the final master metadata file ---
    for record in all_metadata:
        del record["_s3_key"]
        
    metadata_jsonl_content = "\n".join([json.dumps(record, ensure_ascii=False) for record in all_metadata])
    print("\nUploading final metadata.jsonl file...")
    api.upload_file(
        path_or_fileobj=metadata_jsonl_content.encode('utf-8'),
        path_in_repo="metadata.jsonl",
        repo_id=HF_REPO_ID, repo_type="dataset"
    )
        
    print("\n🎉🎉🎉 Dataset finalization and upload complete! 🎉🎉🎉")
    print(f"Check your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    finalize_and_upload()

import os
import json
import tqdm
from pathlib import Path
from huggingface_hub import HfApi
import boto3
import tempfile
import tarfile
import math

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "yt-pipeline-bucket")
S3_PREFIX = "processed/"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "leo-fixie/videos-test")
HF_TOKEN = os.environ.get("HF_TOKEN")
# New: Define how many audio files go into each .tar archive
FILES_PER_TAR_BATCH = 5000 

def create_and_upload_tar_batch(api: HfApi, s3_client, batch_records: list, batch_num: int):
    """
    Downloads a batch of files from S3, creates a .tar archive, and uploads it to HF.
    """
    batch_name = f"batch_{batch_num:05d}.tar"
    hf_tar_path = f"archives/{batch_name}"
    print(f"\nProcessing {batch_name} with {len(batch_records)} files...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_path = temp_path / "audio"
        os.makedirs(download_path)

        # Step 1: Download all files for this batch from S3
        for record in tqdm.tqdm(batch_records, desc=f"Downloading for {batch_name}"):
            s3_key = record["_s3_key"]
            # The filename within the tar will not have the 'audio/' prefix
            local_filename = s3_key.split('/')[-1]
            s3_client.download_file(S3_BUCKET_NAME, s3_key, str(download_path / local_filename))

        # Step 2: Create the .tar archive
        tar_filepath = temp_path / batch_name
        print(f"  Creating archive: {tar_filepath}")
        with tarfile.open(tar_filepath, "w") as tar:
            # The arcname parameter sets the path inside the tar file
            tar.add(str(download_path), arcname="audio")
        
        # Step 3: Upload the single .tar file to Hugging Face
        print(f"  Uploading {batch_name} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=str(tar_filepath),
            path_in_repo=hf_tar_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"  ✅ Successfully uploaded {batch_name}.")
        # The temporary directory and its contents are automatically deleted

def finalize_and_upload():
    """
    Scans S3, aggregates metadata, creates batched .tar archives of audio files,
    and uploads everything to Hugging Face.
    """
    s3_client = boto3.client('s3')
    api = HfApi(token=HF_TOKEN)
    
    print(f"🚀 Ensuring repository '{HF_REPO_ID}' exists on Hugging Face Hub...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    print(f"🔍 Scanning s3://{S3_BUCKET_NAME}/{S3_PREFIX} for processed videos...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX, Delimiter='/')
    video_prefixes = [p['Prefix'] for p in pages.search('CommonPrefixes') if p]
    
    print(f"🔍 Found {len(video_prefixes)} processed video prefixes.")
    all_metadata = []

    # --- 1. Aggregate All Metadata from S3 First ---
    print("📝 Aggregating all metadata from S3...")
    for s3_video_prefix in tqdm.tqdm(video_prefixes, desc="Scanning metadata"):
        video_id = s3_video_prefix.split('/')[-2]
        metadata_s3_key = f"{s3_video_prefix}all_segments.json"
        
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
            segments = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            continue
        
        for i, segment in enumerate(segments):
            mp3_filename = f"{video_id}_{i}.mp3"
            # This path is now a "virtual" path. The data loader will know to look inside the tars for it.
            hf_audio_path = f"audio/{mp3_filename}"
            s3_audio_key = f"{s3_video_prefix}{mp3_filename}"
            
            new_record = {
                "id": f"{video_id}_{i}", "audio": hf_audio_path, "text": segment.get("text", ""),
                "speaker_id": segment.get("speaker", "UNKNOWN"), "duration": segment.get("end", 0) - segment.get("start", 0),
                "dnsmos": segment.get("dnsmos", 0.0), "language": segment.get("language", "UNKNOWN"),
                "_s3_key": s3_audio_key # We'll use this to find the file in S3
            }
            all_metadata.append(new_record)

    if not all_metadata:
        print("[!] No valid segments found to create a dataset.")
        return

    # --- 2. Upload Audio in Batched Tar Archives ---
    num_batches = math.ceil(len(all_metadata) / FILES_PER_TAR_BATCH)
    print(f"\nTotal audio files to process: {len(all_metadata)}")
    print(f"Creating {num_batches} .tar batches of {FILES_PER_TAR_BATCH} files each.")

    for i in range(num_batches):
        start_index = i * FILES_PER_TAR_BATCH
        end_index = start_index + FILES_PER_TAR_BATCH
        batch_records = all_metadata[start_index:end_index]
        create_and_upload_tar_batch(api, s3_client, batch_records, i)

    # --- 3. Upload the final master metadata file ---
    # We remove the temporary _s3_key before saving the final metadata
    for record in all_metadata:
        del record["_s3_key"]
        
    metadata_jsonl_content = "\n".join([json.dumps(record) for record in all_metadata])
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

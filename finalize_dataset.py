import os
import json
import tqdm
from pathlib import Path
from huggingface_hub import HfApi, HfFolder
import boto3
import io

# --- Configuration ---
S3_BUCKET_NAME = "yt-pipeline-bucket"
S3_PREFIX = "processed/"
HF_REPO_ID = "leo-fixie/conversational-podcasts"
HF_TOKEN = os.getenv("HF_TOKEN")

def finalize_and_upload():
    """
    Scans an S3 prefix, aggregates metadata, and uploads the dataset 
    to Hugging Face in a normalized, flat format.
    """
    s3_client = boto3.client('s3')
    
    # --- 1. List all video "directories" in S3 ---
    print(f"🔍 Scanning s3://{S3_BUCKET_NAME}/{S3_PREFIX} for processed videos...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX, Delimiter='/')
    video_prefixes = [p['Prefix'] for p in pages.search('CommonPrefixes') if p]
    
    print(f"🔍 Found {len(video_prefixes)} processed video prefixes.")

    all_metadata = []

    # --- 2. Aggregate Metadata from S3 ---
    print("📝 Aggregating metadata from S3...")
    for s3_video_prefix in tqdm.tqdm(video_prefixes, desc="Scanning videos"):
        video_id = s3_video_prefix.split('/')[-2]
        metadata_s3_key = f"{s3_video_prefix}all_segments.json"
        
        try:
            # Download the metadata file from S3 into memory
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=metadata_s3_key)
            segments = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            print(f"  [Warning] Metadata file not found for {video_id}, skipping.")
            continue
        
        for i, segment in enumerate(segments):
            mp3_filename = f"{video_id}_{i}.mp3"
            hf_audio_path = f"audio/{mp3_filename}"
            s3_audio_key = f"{s3_video_prefix}{mp3_filename}"
            
            new_record = {
                "id": f"{video_id}_{i}", "audio": hf_audio_path, "text": segment.get("text", ""),
                "speaker_id": segment.get("speaker", "UNKNOWN"), "duration": segment.get("end", 0) - segment.get("start", 0),
                "dnsmos": segment.get("dnsmos", 0.0), "language": segment.get("language", "UNKNOWN"),
                "_s3_key": s3_audio_key # Internal key to find the audio in S3 later
            }
            all_metadata.append(new_record)

    if not all_metadata:
        print("[!] No valid segments found to create a dataset.")
        return

    # --- 3. Write and upload the master metadata file ---
    metadata_jsonl_content = "\n".join([json.dumps(record) for record in all_metadata])
    print(f"\n🚀 Preparing to upload to Hugging Face repository: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    print("Uploading metadata.jsonl...")
    api.upload_file(
        path_or_fileobj=metadata_jsonl_content.encode('utf-8'),
        path_in_repo="metadata.jsonl",
        repo_id=HF_REPO_ID, repo_type="dataset"
    )

    # --- 4. Upload all audio files by streaming from S3 ---
    print(f"Uploading {len(all_metadata)} audio files by streaming from S3...")
    for record in tqdm.tqdm(all_metadata, desc="Uploading audio"):
        s3_key = record["_s3_key"]
        hf_path = record["audio"]
        
        # Create a file-like object by streaming the S3 content
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        s3_file_stream = io.BytesIO(response['Body'].read())
        
        api.upload_file(
            path_or_fileobj=s3_file_stream,
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type="dataset"
        )
        
    print("\n🎉🎉🎉 Dataset finalization and upload complete! 🎉🎉🎉")
    print(f"Check your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    finalize_and_upload()

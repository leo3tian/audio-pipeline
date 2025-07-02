import os
import boto3
from yt_dlp import YoutubeDL
import tqdm

# --- Configuration ---
S3_BUCKET = "yt-pipeline-bucket"
# This is the S3 "folder" where the initial video tasks will be created.
S3_TASKS_PREFIX = "tasks/videos_todo/"
# The list of YouTube channel URLs you want to process
CHANNEL_URLS = [
    "https://www.youtube.com/@CodeAesthetic/videos"
]

def main():
    """
    Scans YouTube channels to find all video URLs and creates a task file in S3
    for each individual video. This only needs to be run once to kick off the pipeline.
    """
    print("0: Starting task setup: fetching all video URLs from channels...")
    s3_client = boto3.client("s3")
    all_video_urls = {} # Use a dict to automatically handle duplicates

    # Use yt-dlp to extract all video URLs from the channels
    ydl_opts = {'extract_flat': True, 'quiet': True, 'ignoreerrors': True}
    with YoutubeDL(ydl_opts) as ydl:
        for channel_url in CHANNEL_URLS:
            print(f"  Fetching videos from: {channel_url}")
            try:
                info = ydl.extract_info(channel_url, download=False)
                if info and info.get("entries"):
                    for entry in info['entries']:
                        if entry and 'id' in entry:
                            video_id = entry['id']
                            full_url = f"https://www.youtube.com/watch?v={video_id}"
                            all_video_urls[video_id] = full_url
            except Exception as e:
                 print(f"  [!] Could not process channel {channel_url}: {e}")

    video_count = len(all_video_urls)
    if video_count == 0:
        print("\n[!] No videos found. Aborting task setup.")
        return

    print(f"\nFound a total of {video_count} unique videos.")
    print(f"Creating task files in s3://{S3_BUCKET}/{S3_TASKS_PREFIX}")
    
    for video_id, url in tqdm.tqdm(all_video_urls.items(), desc="Creating tasks"):
        task_key = f"{S3_TASKS_PREFIX}{video_id}.task"
        
        # The content of the task file is simply the video URL
        s3_client.put_object(Bucket=S3_BUCKET, Key=task_key, Body=url)
            
    print("\n✅ Video task setup complete.")
    print(f"   {video_count} tasks created in s3://{S3_BUCKET}/{S3_TASKS_PREFIX}")

if __name__ == "__main__":
    main()

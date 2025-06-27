import os
import subprocess
from streaming import MDSWriter
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor
import time
import random

# --- Configuration ---
CHANNEL_URLS = [
    "https://www.youtube.com/@CodeAesthetic",
    # Add other channels to process here...
]

S3_OUTPUT_DIR = os.environ.get("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset") 
SAMPLE_RATE = 24000
NUM_WORKERS = 16

# Define the columns (schema) for the raw audio dataset
DATASET_COLUMNS = {
    'video_id': 'str',
    'audio': 'bytes', # The raw, converted WAV audio bytes
    'sample_rate': 'int'
}

# --- yt-dlp Configuration ---
# Updated to use a cookies file for authentication
YDL_OPTS = {
    'format': 'bestaudio/best',
    'quiet': True,
    'ignoreerrors': True,
    'cookiefile': '/home/ec2-user/cookies.txt' # Points to the cookies file on the EC2 instance
}

def download_and_write_to_stream(video_url: str, writer: MDSWriter):
    """
    Gets a direct audio URL, pipes it through ffmpeg for on-the-fly conversion
    to WAV format, and writes the resulting audio bytes to the MDSWriter.
    This method avoids saving temporary audio files to disk.
    """
    try:
        # Step 1: Get metadata and the direct URL of the best audio stream
        print(f"Fetching metadata for: {video_url}")
        with YoutubeDL(YDL_OPTS) as ydl:
            # The extract_info call will now use the cookies
            info_dict = ydl.extract_info(video_url, download=False)
            if not info_dict:
                raise ValueError("yt-dlp returned no info, possibly due to an error handled internally.")
            
            audio_url = info_dict.get('url')
            video_id = info_dict.get('id', 'unknown')
            if not audio_url:
                raise ValueError("Could not extract direct audio URL.")

        # Step 2: Use ffmpeg to stream and convert the audio in memory
        print(f"  Streaming and converting: {video_id}")
        ffmpeg_command = [
            'ffmpeg',
            '-i', audio_url,
            '-f', 'wav',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-'
        ]
        
        # FIX: Added try/except block to handle ffmpeg crashes gracefully
        try:
            process = subprocess.run(ffmpeg_command, capture_output=True, check=True)
            audio_bytes = process.stdout
            if not audio_bytes:
                raise ValueError("ffmpeg produced no output bytes.")
        except subprocess.CalledProcessError as e:
            # This will catch ffmpeg crashes (like the SIGSEGV error)
            error_output = e.stderr.decode() if e.stderr else "No stderr output"
            raise RuntimeError(f"ffmpeg failed with exit code {e.returncode}: {error_output}")

        # Step 3: Write the in-memory audio bytes to the streaming dataset
        sample = {
            'video_id': video_id,
            'audio': audio_bytes,
            'sample_rate': SAMPLE_RATE
        }
        writer.write(sample)
        print(f"  ✅ Wrote {video_id} to stream.")

        # Add a small random delay to mimic human behavior
        time.sleep(random.uniform(1, 3))
            
    except Exception as e:
        print(f"[ERROR] Failed to process {video_url}: {e}")

def main():
    """
    Collects all video URLs and processes them in parallel, writing the
    final output to a MosaicML StreamingDataset on S3.
    """
    # 1. Collect all video URLs from all channels
    print("Collecting all video URLs...")
    all_video_urls = []
    flat_opts = {'extract_flat': True, 'quiet': True, 'ignoreerrors': True, 'cookiefile': YDL_OPTS['cookiefile']}
    with YoutubeDL(flat_opts) as ydl:
        for channel_url in CHANNEL_URLS:
            try:
                info = ydl.extract_info(channel_url, download=False)
                if info and info.get("entries"):
                    urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info['entries'] if e and 'id' in e]
                    all_video_urls.extend(urls)
            except Exception as e:
                 print(f"Could not process channel {channel_url}: {e}")
    
    print(f"\nFound a total of {len(all_video_urls)} videos to process.")

    # 2. Use MDSWriter to create the dataset on S3
    with MDSWriter(out=S3_OUTPUT_DIR, columns=DATASET_COLUMNS) as writer:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(download_and_write_to_stream, url, writer) for url in all_video_urls]
            for future in futures:
                future.result()

    print("\n✅ Successfully created the StreamingDataset on S3.")
    print(f"   Your dataset is located at: {S3_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

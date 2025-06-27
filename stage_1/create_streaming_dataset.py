import os
import subprocess
from streaming import MDSWriter
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
CHANNEL_URLS = [
    "https://www.youtube.com/@CodeAesthetic",
    # Add other channels to process here...
]

# This will be read from an environment variable set in the EC2 User Data
S3_OUTPUT_DIR = os.environ.get("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset") 
SAMPLE_RATE = 24000
# Number of parallel download/conversion threads
NUM_WORKERS = 16

# Define the columns (schema) for the raw audio dataset
DATASET_COLUMNS = {
    'video_id': 'str',
    'audio': 'bytes', # The raw, converted WAV audio bytes
    'sample_rate': 'int'
}

# yt-dlp options to get a direct stream URL without downloading
YDL_OPTS = {
    'format': 'bestaudio/best',
    'quiet': True,
    'ignoreerrors': True,
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
            info_dict = ydl.extract_info(video_url, download=False)
            audio_url = info_dict.get('url')
            video_id = info_dict.get('id', 'unknown')
            if not audio_url:
                raise ValueError("Could not extract direct audio URL.")

        # Step 2: Use ffmpeg to stream and convert the audio in memory
        print(f"  Streaming and converting: {video_id}")
        ffmpeg_command = [
            'ffmpeg',
            '-i', audio_url,      # Input from the direct stream URL
            '-f', 'wav',          # Output format is WAV
            '-ar', str(SAMPLE_RATE), # Resample audio
            '-ac', '1',           # Convert to mono
            '-',                  # Send output to stdout
        ]
        
        # Run the command and capture the output bytes
        process = subprocess.run(ffmpeg_command, capture_output=True, check=True)
        audio_bytes = process.stdout
        
        if not audio_bytes:
            raise ValueError("ffmpeg produced no output bytes.")

        # Step 3: Write the in-memory audio bytes to the streaming dataset
        sample = {
            'video_id': video_id,
            'audio': audio_bytes,
            'sample_rate': SAMPLE_RATE
        }
        writer.write(sample)
        print(f"  ✅ Wrote {video_id} to stream.")
            
    except Exception as e:
        # We will log the error but not stop the entire process
        print(f"[ERROR] Failed to process {video_url}: {e}")

def main():
    """
    Collects all video URLs and processes them in parallel, writing the
    final output to a MosaicML StreamingDataset on S3.
    """
    # 1. Collect all video URLs from all channels
    print("Collecting all video URLs...")
    all_video_urls = []
    flat_opts = {'extract_flat': True, 'quiet': True, 'ignoreerrors': True}
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
        # Use a ThreadPoolExecutor to download/convert in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all download tasks
            futures = [executor.submit(download_and_write_to_stream, url, writer) for url in all_video_urls]
            
            # This loop just waits for all threads to complete
            for future in futures:
                future.result()

    print("\n✅ Successfully created the StreamingDataset on S3.")
    print(f"   Your dataset is located at: {S3_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

import os
import subprocess
import tempfile
from pathlib import Path
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

S3_OUTPUT_DIR = os.environ.get("S3_OUTPUT_DIR", "s3://your-bucket-name-here/streaming_dataset") 
SAMPLE_RATE = 24000
NUM_WORKERS = 16

# Define the columns (schema) for the raw audio dataset
DATASET_COLUMNS = {
    'video_id': 'str',
    'audio': 'bytes', # The raw, converted WAV audio bytes
    'sample_rate': 'int'
}

def download_and_write_to_stream(video_url: str, writer: MDSWriter):
    """
    Downloads an audio file to a temporary local disk, converts it using ffmpeg,
    and writes the resulting audio bytes to the MDSWriter. This method is more stable
    than in-memory streaming for ffmpeg.
    """
    # Create a unique temporary directory for this download
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_path = Path(temp_dir)
            
            # --- yt-dlp Configuration to download to our temp directory ---
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'ignoreerrors': True,
                'cookiefile': '/home/ec2-user/cookies.txt',
                # Set the output template to save inside our temp dir
                'outtmpl': str(temp_path / '%(id)s.%(ext)s'),
            }

            # Step 1: Download the full audio file to the local disk
            print(f"Downloading: {video_url}")
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(video_url, download=True)
                if not info_dict:
                    raise ValueError("yt-dlp returned no info.")
                
                video_id = info_dict.get('id', 'unknown')
                # Find the path to the downloaded file
                downloaded_filepath = ydl.prepare_filename(info_dict)

                if not os.path.exists(downloaded_filepath):
                     raise FileNotFoundError(f"Could not find downloaded file at {downloaded_filepath}")

            # Step 2: Use ffmpeg to convert the local file
            print(f"  Converting: {video_id}")
            output_wav_bytes = convert_audio_to_wav(downloaded_filepath)

            # Step 3: Write the audio bytes to the streaming dataset
            sample = {
                'video_id': video_id,
                'audio': output_wav_bytes,
                'sample_rate': SAMPLE_RATE
            }
            writer.write(sample)
            print(f"  ✅ Wrote {video_id} to stream.")

            # Add a small random delay to mimic human behavior
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"[ERROR] Failed to process {video_url}: {e}")
        # The 'with tempfile.TemporaryDirectory()' block automatically cleans up the temp_dir

def convert_audio_to_wav(local_filepath: str) -> bytes:
    """
    Converts a local audio file to WAV format in memory using ffmpeg.
    """
    ffmpeg_command = [
        'ffmpeg',
        '-i', local_filepath,   # Input from the stable local file
        '-f', 'wav',            # Output format is WAV
        '-ar', str(SAMPLE_RATE),   # Resample to our target sample rate
        '-ac', '1',             # Convert to mono
        '-'                     # Output to stdout
    ]
    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, check=True)
        return process.stdout
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode() if e.stderr else "No stderr output"
        raise RuntimeError(f"ffmpeg failed with exit code {e.returncode} for file {local_filepath}: {error_output}")


def main():
    """
    Collects all video URLs and processes them in parallel, writing the
    final output to a MosaicML StreamingDataset on S3.
    """
    # 1. Collect all video URLs from all channels
    print("Collecting all video URLs...")
    all_video_urls = []
    # We still need cookies for the initial URL collection
    cookie_path = '/home/ec2-user/cookies.txt'
    flat_opts = {'extract_flat': True, 'quiet': True, 'ignoreerrors': True, 'cookiefile': cookie_path}
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

import os
import subprocess
import tempfile
from pathlib import Path
from streaming import MDSWriter
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading
import itertools
import shutil
import tqdm

# --- Configuration ---
CHANNEL_URLS = [
    "https://www.youtube.com/@CodeAesthetic/videos",
]

S3_OUTPUT_DIR = os.environ.get("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset") 
SAMPLE_RATE = 24000
NUM_WORKERS = 4
FAILED_URLS_LOG = "failed_urls.log"
COOLDOWN_PERIOD_SECONDS = 60

# --- Synchronization objects for rate-limiting cooldown ---
cooldown_event = threading.Event()
cooldown_event.set()
cooldown_lock = threading.Lock()

# Define the columns (schema) for the raw audio dataset
DATASET_COLUMNS = {
    'video_id': 'str',
    'audio': 'bytes',
    'sample_rate': 'int'
}

def download_and_write_to_stream(video_url: str, writer: MDSWriter, task_num: int, temp_dir: Path):
    """
    Downloads and converts a single video's audio, writing it to the stream.
    Now uses a shared temporary directory to avoid race conditions.
    """
    cooldown_event.wait()
    
    # Use a unique subdirectory within the shared temp_dir for this specific task
    task_temp_path = temp_dir / f"task_{task_num}"
    os.makedirs(task_temp_path, exist_ok=True)

    try:
        ydl_opts = {
            'format': 'bestaudio/best', 'quiet': True, 'ignoreerrors': True,
            'cookiefile': '/home/ec2-user/cookies.txt',
            'outtmpl': str(task_temp_path / '%(id)s.%(ext)s'),
        }

        # Step 1: Download the full audio file
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            if not info_dict:
                raise ValueError("yt-dlp returned no info, likely due to a download error.")
            
            video_id = info_dict.get('id', 'unknown')
            downloaded_filepath = ydl.prepare_filename(info_dict)

            if not os.path.exists(downloaded_filepath):
                 raise FileNotFoundError(f"Could not find downloaded file at {downloaded_filepath}")

        # Step 2: Convert the local file using ffmpeg
        output_wav_bytes = convert_audio_to_wav(downloaded_filepath)

        # Step 3: Write to the streaming dataset
        sample = {
            'video_id': video_id,
            'audio': output_wav_bytes,
            'sample_rate': SAMPLE_RATE
        }
        writer.write(sample)
        print(f"  ✅ [Video #{task_num}] Wrote {video_id} to stream.")

        time.sleep(random.uniform(1, 3))
    finally:
        # Clean up the specific task's subdirectory
        shutil.rmtree(task_temp_path, ignore_errors=True)


def convert_audio_to_wav(local_filepath: str) -> bytes:
    """Converts a local audio file to WAV format in memory using ffmpeg."""
    ffmpeg_command = [
        'ffmpeg', '-i', local_filepath, '-f', 'wav',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-'
    ]
    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, check=True, timeout=300)
        return process.stdout
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode() if e.stderr else "No stderr output"
        raise RuntimeError(f"ffmpeg failed with exit code {e.returncode}: {error_output}")

def main():
    """
    Main orchestration function. It now handles thread failures gracefully
    and logs problematic URLs.
    """
    # 1. Collect all video URLs
    print("Collecting all video URLs...")
    all_video_urls = []
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
    
    # FIX: Create a single parent temporary directory for the entire run
    main_temp_dir = tempfile.mkdtemp(prefix="streaming_parent_")
    try:
        with open(FAILED_URLS_LOG, "w") as f_failures:
            with MDSWriter(out=S3_OUTPUT_DIR, columns=DATASET_COLUMNS) as writer:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    
                    task_counter = itertools.count(start=1)
                    future_to_task = {}
                    for url in all_video_urls:
                        task_num = next(task_counter)
                        # Pass the shared temp directory path to each thread
                        future = executor.submit(download_and_write_to_stream, url, writer, task_num, Path(main_temp_dir))
                        future_to_task[future] = (url, task_num)
                    
                    for future in tqdm.tqdm(as_completed(future_to_task), total=len(all_video_urls), desc="Processing Videos"):
                        url, task_num = future_to_task[future]
                        try:
                            future.result()
                        except Exception as exc:
                            print(f'\n[!!!] Task #{task_num} for URL {url} generated an exception: {exc}')
                            f_failures.write(f'{url}\n')

                            with cooldown_lock:
                                if cooldown_event.is_set():
                                    print(f"\n[!!!] Detected a potential rate limit. Initiating {COOLDOWN_PERIOD_SECONDS}-second cooldown...")
                                    cooldown_event.clear()
                                    time.sleep(COOLDOWN_PERIOD_SECONDS)
                                    print("[OK] Cooldown finished. Resuming downloads.")
                                    cooldown_event.set()
    finally:
        # Clean up the main temporary directory at the very end
        shutil.rmtree(main_temp_dir, ignore_errors=True)


    print("\n✅ Processing complete.")
    print(f"   StreamingDataset created at: {S3_OUTPUT_DIR}")
    print(f"   A log of any failed URLs has been saved to: {FAILED_URLS_LOG}")

if __name__ == "__main__":
    main()

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
import queue

# --- Configuration ---
CHANNEL_URLS = [
    "https://www.youtube.com/@mostlynitpicking/videos",
]

S3_OUTPUT_DIR = os.environ.get("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset") 
SAMPLE_RATE = 24000
NUM_WORKERS = 4
FAILED_URLS_LOG = "failed_urls.log"

# --- Producer-Consumer Queue ---
# This queue will safely pass data from the many downloader threads to the single writer thread.
data_queue = queue.Queue(maxsize=NUM_WORKERS * 2)

def producer_thread(video_url: str, task_num: int, temp_dir: Path):
    """
    Producer: Downloads a video, processes it, and puts the result into the data_queue.
    This function no longer interacts with the MDSWriter.
    """
    task_temp_path = temp_dir / f"task_{task_num}"
    os.makedirs(task_temp_path, exist_ok=True)

    try:
        ydl_opts = {
            'format': 'bestaudio/best', 'quiet': True, 'ignoreerrors': True,
            'cookiefile': '/home/ec2-user/cookies.txt',
            'outtmpl': str(task_temp_path / '%(id)s.%(ext)s'),
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            if not info_dict:
                raise ValueError("yt-dlp returned no info")
            
            video_id = info_dict.get('id', 'unknown')
            downloaded_filepath = ydl.prepare_filename(info_dict)

            if not os.path.exists(downloaded_filepath):
                 raise FileNotFoundError(f"Could not find downloaded file at {downloaded_filepath}")

        output_wav_bytes = convert_audio_to_wav(downloaded_filepath)

        sample = {
            'video_id': video_id,
            'audio': output_wav_bytes,
            'sample_rate': SAMPLE_RATE
        }
        # Put the completed sample into the queue for the writer thread to process.
        data_queue.put(sample)
        return None  # Return None on success
    except Exception as e:
        # Return the exception to be handled by the main thread
        return e
    finally:
        shutil.rmtree(task_temp_path, ignore_errors=True)

def consumer_thread(writer: MDSWriter, total_videos: int):
    """
    Consumer: Pulls samples from the data_queue and writes them using the MDSWriter.
    This is the ONLY thread that touches the writer.
    """
    with tqdm.tqdm(total=total_videos, desc="Writing to MDS") as pbar:
        while True:
            sample = data_queue.get()
            # A None in the queue is the signal that all producers are finished.
            if sample is None:
                data_queue.task_done()
                break
            
            try:
                writer.write(sample)
                pbar.update(1)
                print(f"  ✅ Wrote {sample['video_id']} to stream.")
            except Exception as e:
                print(f"\n[!!!] MDSWriter failed to write sample {sample.get('video_id', 'N/A')}: {e}")
            
            data_queue.task_done()

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
    Main orchestration function using a producer-consumer pattern.
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
    
    total_videos = len(all_video_urls)
    print(f"\nFound a total of {total_videos} videos to process.")
    
    # Use a stable directory for all temporary file operations
    main_temp_dir = os.path.abspath("pipeline_temp_dir")
    if os.path.exists(main_temp_dir):
        shutil.rmtree(main_temp_dir)
    os.makedirs(main_temp_dir)
    tempfile.tempdir = main_temp_dir

    try:
        with open(FAILED_URLS_LOG, "w") as f_failures:
            DATASET_COLUMNS = {'video_id': 'str', 'audio': 'bytes', 'sample_rate': 'int'}
            with MDSWriter(out=S3_OUTPUT_DIR, columns=DATASET_COLUMNS) as writer:
                
                # Start the single consumer thread
                writer_thread = threading.Thread(target=consumer_thread, args=(writer, total_videos))
                writer_thread.start()

                # Start the producer threads
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    task_counter = itertools.count(start=1)
                    future_to_url = {
                        executor.submit(producer_thread, url, next(task_counter), Path(main_temp_dir)): url
                        for url in all_video_urls
                    }

                    for future in tqdm.tqdm(as_completed(future_to_url), total=total_videos, desc="Downloading Videos"):
                        url = future_to_url[future]
                        try:
                            # Check for exceptions returned by the producer
                            result = future.result()
                            if isinstance(result, Exception):
                                print(f'\n[!!!] Download failed for URL {url}: {result}')
                                f_failures.write(f'{url}\n')
                        except Exception as exc:
                            print(f'\n[!!!] Critical error in producer thread for URL {url}: {exc}')
                            f_failures.write(f'{url}\n')

                # All producers are done, signal the consumer to finish
                data_queue.put(None)
                
                # Wait for the consumer thread to finish writing all queued items
                writer_thread.join()

    finally:
        shutil.rmtree(main_temp_dir, ignore_errors=True)

    print("\n✅ Processing complete.")
    print(f"   StreamingDataset created at: {S3_OUTPUT_DIR}")
    print(f"   A log of any failed URLs has been saved to: {FAILED_URLS_LOG}")

if __name__ == "__main__":
    main()

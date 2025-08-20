# reqs:
# pip install yt-dlp boto3 curl-cffi
import os
import multiprocessing
import tempfile
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from yt_dlp import YoutubeDL
import time
import json
import subprocess
import shutil
import random
import logging
from datetime import datetime

# --- Configuration ---
S3_BUCKET = "youtube-dataset-west"
# The S3 "folder" where the final raw audio files will be stored
S3_RAW_AUDIO_PREFIX = "raw_audio/"

# SQS Configuration
SQS_QUEUE_URL = os.getenv("AWS_QUEUE_DOWNLOAD")
if not SQS_QUEUE_URL:
    raise ValueError("Environment variable AWS_QUEUE_DOWNLOAD must be set")

# Worker Configuration
NUM_WORKERS = 1
MAX_CONSECUTIVE_FAILURES = 5
# How long to wait for new messages (long polling)
SQS_WAIT_TIME = 20
# Maximum number of messages to receive at once
SQS_BATCH_SIZE = 1

# Download speed configuration (in bytes/s)
MAX_DOWNLOAD_SPEED = 6_250_000  # 50 Mbps
MIN_DOWNLOAD_SPEED = 4_000_000  # 32 Mbps
# Rest period between videos (in seconds)
MIN_REST_PERIOD = 30
MAX_REST_PERIOD = 60

# Logging Configuration
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class DownloadTracker:
    def __init__(self, logger):
        self.start_time = None
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.logger = logger
        self.speeds = []
        
    def progress_hook(self, d):
        if d['status'] == 'downloading':
            if not self.start_time:
                self.start_time = time.time()
            
            # Track download progress
            self.downloaded_bytes = d.get('downloaded_bytes', 0)
            self.total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            
            # Calculate current speed
            speed = d.get('speed', 0)
            if speed:
                self.speeds.append(speed)
        
        elif d['status'] == 'finished':
            if self.start_time and self.speeds:
                duration = time.time() - self.start_time
                avg_speed = sum(self.speeds) / len(self.speeds)
                max_speed = max(self.speeds)
                
                # Convert to MB/s for readability
                avg_speed_mb = avg_speed / (1024 * 1024)
                max_speed_mb = max_speed / (1024 * 1024)
                
                self.logger.info(
                    f"📊 Download stats: {duration:.1f}s, "
                    f"avg: {avg_speed_mb:.1f} MB/s, "
                    f"max: {max_speed_mb:.1f} MB/s"
                )

def check_dependencies():
    """Check if required command-line tools are available."""
    missing = []
    
    # Check for ffmpeg
    if not shutil.which('ffmpeg'):
        missing.append('ffmpeg')
    
    if missing:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(missing)}. "
            "Please install them before running this script.\n"
            "On Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "On macOS: brew install ffmpeg\n"
            "On Windows: choco install ffmpeg"
        )

def convert_to_m4a(input_path: Path, output_path: Path) -> bool:
    """
    Convert audio file to M4A format using ffmpeg.
    Returns True if conversion was successful.
    """
    try:
        # Use ffmpeg to convert to M4A with AAC codec
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', str(input_path),  # Input file
            '-c:a', 'aac',  # Use AAC codec
            '-b:a', '128k',  # Set bitrate to 128kbps
            '-movflags', '+faststart',  # Optimize for streaming
            str(output_path)  # Output file
        ]
        
        # Run ffmpeg silently unless there's an error
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  FFmpeg conversion failed: {result.stderr}")
            return False
            
        return output_path.exists()
        
    except Exception as e:
        print(f"  Conversion error: {str(e)}")
        return False

def receive_task(sqs_client):
    """
    Receives a task from the SQS queue.
    Returns None if no tasks are available.
    """
    try:
        response = sqs_client.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=SQS_BATCH_SIZE,
            WaitTimeSeconds=SQS_WAIT_TIME,
            AttributeNames=['All']
        )
        
        if 'Messages' not in response:
            return None
            
        message = response['Messages'][0]  # We only asked for 1 message
        try:
            body = json.loads(message['Body'])
            return {
                'receipt_handle': message['ReceiptHandle'],
                'video_id': body['video_id'],
                'url': body['url']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [!] Invalid message format: {e}")
            # Delete malformed message so it doesn't block the queue
            sqs_client.delete_message(
                QueueUrl=SQS_QUEUE_URL,
                ReceiptHandle=message['ReceiptHandle']
            )
            return None
            
    except Exception as e:
        print(f"  [!] Error receiving message: {e}")
        return None

def delete_task(sqs_client, receipt_handle):
    """
    Deletes a completed task from the SQS queue.
    """
    try:
        sqs_client.delete_message(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )
        return True
    except Exception as e:
        print(f"  [!] Failed to delete message: {e}")
        return False

def setup_logging(worker_rank: int):
    """Setup logging for a worker process."""
    # Create a logger for this worker
    logger = logging.getLogger(f"worker-{worker_rank}")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    # Console handler - only INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(message)s')
    console.setFormatter(console_fmt)
    
    # File handler - DEBUG and above, with timestamps and more detail
    log_file = LOG_DIR / f"worker_{worker_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_fmt)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger

def random_rest():
    """Take a random rest between videos to avoid patterns."""
    return random.uniform(MIN_REST_PERIOD, MAX_REST_PERIOD)

def download_audio(video_url: str, temp_dir: Path, worker_rank: int, logger: logging.Logger):
    """Downloads a single video's audio and ensures it's in M4A format."""
    tracker = DownloadTracker(logger)
    
    ydl_opts = {
        # --- Format Selection ---
        'format': 'bestaudio[abr<=128]/bestaudio',  # Allow any audio format
        'outtmpl': str(temp_dir / '%(id)s.%(ext)s'),
        
        # --- Browser Impersonation & Headers ---
        'http_headers': {
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
        },
        'impersonate_target': 'chrome',  # Latest Chrome
        
        # --- Download Settings ---
        'retries': 3,              # Reduced retries
        'fragment_retries': 3,     # Reduced fragment retries
        'skip_unavailable_fragments': True,
        'keepvideo': False,        # Delete video after audio extraction
        
        # --- Rate Limiting ---
        'ratelimit': MAX_DOWNLOAD_SPEED,      # 50 Mbps max
        'throttledratelimit': MIN_DOWNLOAD_SPEED,  # Allow drops to 32 Mbps
        
        # --- Error Handling ---
        'ignoreerrors': True,      # Skip unavailable videos
        'verbose': False,          # Hide debug info
        'quiet': True,  # Suppress yt-dlp's output
        
        # --- Progress Monitoring ---
        'progress_hooks': [tracker.progress_hook],
        
        # --- Misc ---
        'no_playlist': True,       # Don't download playlists by accident
        'extract_audio': True,     # Extract audio
        'postprocessor_args': [
            '-ar', '44100',        # Sample rate
            '-ac', '2',            # Stereo
            '-b:a', '128k',        # Bitrate
        ],
        
        # --- Additional Optimizations ---
        'socket_timeout': 20,      # Socket timeout
        'retry_sleep_functions': False,  # Don't sleep between retries
        'file_access_retries': 3,  # Reduce file access retries
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            logger.info(f"⏬ Downloading {video_url}")
            info = ydl.extract_info(video_url, download=True)
            if not info:
                raise ValueError("yt-dlp returned no info")
            
            video_id = info['id']
            downloaded_path = Path(ydl.prepare_filename(info))
            
            if not downloaded_path.exists():
                raise FileNotFoundError(f"Download failed - file not found: {downloaded_path}")
            
            # Ensure we have an M4A file
            output_path = temp_dir / f"{video_id}.m4a"
            if downloaded_path.suffix != '.m4a':
                logger.debug(f"Converting {downloaded_path.suffix} to M4A...")
                if not convert_to_m4a(downloaded_path, output_path):
                    raise RuntimeError(f"Failed to convert {downloaded_path} to M4A")
                downloaded_path.unlink()  # Remove the original file
            else:
                # If it's already M4A, just rename it
                downloaded_path.rename(output_path)
            
            if not output_path.exists():
                raise FileNotFoundError(f"Final M4A file not found: {output_path}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Download failed: {type(e).__name__}: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            raise

def downloader_worker(rank: int, failure_counter):
    """A worker process that continuously claims and processes video download tasks."""
    logger = setup_logging(rank)
    s3_client = boto3.client('s3')
    sqs_client = boto3.client('sqs')
    logger.info(f"Worker-{rank} started")
    
    while True:
        if failure_counter.value >= MAX_CONSECUTIVE_FAILURES:
            logger.error(f"Max failures reached ({MAX_CONSECUTIVE_FAILURES}). Exiting.")
            break

        task = receive_task(sqs_client)
        if not task:
            logger.info("No more tasks found. Exiting.")
            break

        video_id = task['video_id']
        video_url = task['url']

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = download_audio(video_url, Path(temp_dir), rank, logger)
                
                s3_key = f"{S3_RAW_AUDIO_PREFIX}{video_id}.m4a"
                logger.debug(f"Uploading to S3: {s3_key}")
                s3_client.upload_file(str(audio_path), S3_BUCKET, s3_key)
            
            if delete_task(sqs_client, task['receipt_handle']):
                logger.info(f"✅ Finished {video_id}")
                failure_counter.value = 0
                
                rest_time = random_rest()
                logger.info(f"😴 Resting for {rest_time:.1f}s...")
                time.sleep(rest_time)
            else:
                logger.warning(f"Task completed but failed to delete message for: {video_id}")
                failure_counter.value += 1

        except Exception as e:
            logger.error(f"CRITICAL FAILURE on {video_id}: {type(e).__name__}: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            
            failure_counter.value += 1
            backoff = min(2 ** (failure_counter.value - 1), 5)
            logger.info(f"Backing off for {backoff:.1f}s...")
            time.sleep(backoff)

def main():
    """Orchestrates the pool of downloader worker processes."""
    try:
        check_dependencies()
    except RuntimeError as e:
        print(f"\n❌ {str(e)}")
        return

    manager = multiprocessing.Manager()
    failure_counter = manager.Value('i', 0)
    
    processes = []
    print(f"🚀 Starting {NUM_WORKERS} worker(s)...")
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=downloader_worker, args=(i, failure_counter))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print("\n✅ All workers finished.")
    print(f"    Total failures: {failure_counter.value}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
import os
import subprocess
import multiprocessing
import threading
import queue # Using the thread-safe queue for internal communication
import tempfile
from pathlib import Path
from yt_dlp import YoutubeDL
import boto3

# --- Configuration ---
URL_LIST_FILE = "master_url_list.txt" 
SAMPLE_RATE = 24000
EMILIA_WORKERS = 6 
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "your-bucket-name-here")

# --- CORE PROCESSING FUNCTIONS (UNCHANGED) ---

def run_emilia_pipe(input_wav_file: str, output_dir: str, device: str):
    """Runs the Emilia-pipe on a specific audio file using a specific GPU."""
    print(f"GPU {device} - Processing file: {input_wav_file}")
    os.makedirs(output_dir, exist_ok=True)
    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline"
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)
    cmd = f"""
    source {conda_setup} && conda activate {conda_env} && export CUDA_VISIBLE_DEVICES={device} && \
    python {emilia_script} --input_file_path '{input_wav_file}' --config_path '{EMILIA_CONFIG_PATH}' --output_dir '{output_dir}'
    """
    try:
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"GPU {device} - ✅ Successfully processed: {input_wav_file}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        print(f"[!] Emilia error on GPU {device} for {input_wav_file}:\n---\n{error_message}\n---")
        raise

def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix."""
    s3_client = boto3.client("s3")
    for local_file in local_directory.rglob("*"):
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)
    print(f"    Uploaded results to s3://{s3_bucket}/{s3_prefix}")


# --- NEW MULTI-THREADED WORKER LOGIC ---

def downloader_task(urls_to_download: list, ready_queue: queue.Queue):
    """
    This function runs in a separate thread. Its only job is to download
    files and put their info onto the 'ready_queue' for the processor.
    """
    for video_url in urls_to_download:
        # Each download gets its own temporary directory
        temp_dir = tempfile.mkdtemp(prefix="audiopipe_")
        try:
            print(f"Downloader - Starting download for: {video_url}")
            temp_path = Path(temp_dir)
            
            ydl_opts = {
                'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                'postprocessor_args': ['-ar', str(SAMPLE_RATE)], 'outtmpl': str(temp_path / '%(id)s.%(ext)s'),
                'quiet': True, 'ignoreerrors': True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_id = info.get('id', 'unknown_video')

            downloaded_files = list(temp_path.glob("*.wav"))
            if not downloaded_files:
                raise FileNotFoundError(f"yt-dlp failed to download WAV for: {video_url}")
            
            # Put the result on the queue for the processor.
            # This is a tuple containing everything the processor needs.
            ready_queue.put((str(downloaded_files[0]), video_id, temp_dir))
            
        except Exception as e:
            print(f"[!!!] DOWNLOADER FAILURE for URL {video_url}: {e}")
            # Clean up the temp directory if download fails
            if os.path.exists(temp_dir):
                Path(temp_dir).rmdir()

def processor_task(ready_queue: queue.Queue, device: str):
    """
    This function runs in a separate thread, dedicated to the GPU. Its only job
    is to wait for items on the 'ready_queue', process them, and upload them.
    """
    while True:
        try:
            # This will block and wait until an item is available on the queue.
            item = ready_queue.get()
            
            # A 'None' item is a sentinel value that signals the end of the work.
            if item is None:
                break
            
            input_wav_path, video_id, temp_dir = item
            temp_path = Path(temp_dir)
            emilia_output_dir = temp_path / "processed"

            # --- 2. PROCESS ---
            run_emilia_pipe(input_wav_path, str(emilia_output_dir), device)
            
            # --- 3. UPLOAD TO S3 ---
            s3_prefix = f"processed/{video_id}"
            print(f"GPU {device} - Uploading results for {video_id} to S3...")
            upload_directory_to_s3(emilia_output_dir, S3_BUCKET_NAME, s3_prefix)
        
        except Exception as e:
            print(f"[!!!] PROCESSOR FAILURE on GPU {device} for video {video_id}: {e}")
        finally:
            # Clean up the entire temporary directory for the video
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                Path(temp_dir).rmdir()
            # Signal that this task from the queue is finished.
            ready_queue.task_done()

def master_worker_process(urls_for_this_worker: list, device: str):
    """
    This is the main target function for each process (one per GPU).
    It orchestrates the downloader and processor threads.
    """
    print(f"Starting master worker for GPU {device} with {len(urls_for_this_worker)} URLs.")
    # This queue has a max size of 2, acting as a small buffer.
    # The downloader will wait if it gets more than 2 files ahead of the processor.
    ready_to_process_queue = queue.Queue(maxsize=2)

    downloader = threading.Thread(target=downloader_task, args=(urls_for_this_worker, ready_to_process_queue))
    processor = threading.Thread(target=processor_task, args=(ready_to_process_queue, device))

    downloader.start()
    processor.start()

    # Wait for the downloader to finish putting all items on the queue.
    downloader.join()
    
    # Put a sentinel value on the queue to signal the processor to stop.
    ready_to_process_queue.put(None)
    
    # Wait for the processor to finish its last task.
    processor.join()
    print(f"Master worker for GPU {device} has finished.")


# --- MAIN ORCHESTRATION (Similar to before) ---

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        print("[!] nvidia-smi not found. Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """Orchestrates the pool of master worker processes."""
    try:
        with open(URL_LIST_FILE, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"[!] URL file '{URL_LIST_FILE}' is empty or not found.")
            return
    except FileNotFoundError:
        print(f"[!] Error: URL file not found: '{URL_LIST_FILE}'. Run 'collect_urls.py' first.")
        return

    processes = []
    available_devices = get_available_gpus()[:EMILIA_WORKERS]
    if not available_devices:
        print("[!] No GPUs detected. Aborting.")
        return
        
    num_devices = len(available_devices)
    # Split the master URL list into chunks for each worker process
    url_chunks = [urls[i::num_devices] for i in range(num_devices)]
    
    print(f"🚀 Starting {num_devices} master worker processes for {len(urls)} videos...")
    
    for i, device in enumerate(available_devices):
        p = multiprocessing.Process(target=master_worker_process, args=(url_chunks[i], device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ Stage 2 Complete: All videos have been processed and uploaded to S3.")
    print(f"➡️  Intermediate results are in bucket: s3://{S3_BUCKET_NAME}/processed/")
    print("➡️  Next step: Run 'finalize_dataset.py' to create the Hugging Face dataset.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()

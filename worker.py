import os
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
from yt_dlp import YoutubeDL
import boto3 # Import the AWS SDK

# --- Configuration ---
URL_LIST_FILE = "master_url_list.txt" 
SAMPLE_RATE = 24000
EMILIA_WORKERS = 6 
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"
# The name of the S3 bucket you created.
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "yt-pipeline-bucket")


def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix."""
    s3_client = boto3.client("s3")
    for local_file in local_directory.rglob("*"):
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            print(f"    Uploading {local_file.name} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)


def run_emilia_pipe(input_wav_file: str, output_dir: str, device: str):
    """Runs the Emilia-pipe on a specific audio file using a specific GPU."""
    print(f"GPU {device} - Processing file: {input_wav_file}")
    os.makedirs(output_dir, exist_ok=True)
    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline"
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)
    cmd = f"""
    source {conda_setup} && conda activate {conda_env} && export CUDA_VISIBLE_DEVICES={device} && \
    python {emilia_script} --input_file_path '{input_wav_file}' --config_path '{EMILIA_CONFIG_PATH}' --output_dir '{output_dir}' --quiet
    """
    try:
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"GPU {device} - ✅ Successfully processed: {input_wav_file}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        print(f"[!] Emilia error on GPU {device} for {input_wav_file}:\n---\n{error_message}\n---")
        raise

def processing_worker(url_queue: multiprocessing.JoinableQueue, device: str):
    """Worker downloads, processes, and uploads results to S3."""
    while True:
        video_url = url_queue.get()
        if video_url is None:
            url_queue.task_done()
            break
        
        with tempfile.TemporaryDirectory(prefix="audiopipe_") as temp_dir:
            try:
                temp_path = Path(temp_dir)
                emilia_output_dir = temp_path / "processed"

                # --- 1. DOWNLOAD ---
                print(f"GPU {device} - Downloading: {video_url}")
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
                
                input_wav_path = str(downloaded_files[0])

                # --- 2. PROCESS ---
                run_emilia_pipe(input_wav_path, str(emilia_output_dir), device)
                
                # --- 3. UPLOAD TO S3 ---
                s3_prefix = f"processed/{video_id}"
                print(f"GPU {device} - 💾 Uploading results for {video_id} to S3...")
                upload_directory_to_s3(emilia_output_dir, S3_BUCKET_NAME, s3_prefix)
                
            except Exception as e:
                print(f"[!!!] CRITICAL FAILURE on GPU {device} for URL {video_url}: {e}")
            finally:
                url_queue.task_done()

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        print("[!] nvidia-smi not found. Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """Orchestrates the worker pool."""
    try:
        with open(URL_LIST_FILE, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"[!] URL file '{URL_LIST_FILE}' is empty or not found.")
            return
    except FileNotFoundError:
        print(f"[!] Error: URL file not found: '{URL_LIST_FILE}'. Run 'collect_urls.py' first.")
        return

    url_queue = multiprocessing.JoinableQueue()
    for url in urls:
        url_queue.put(url)

    processes = []
    available_devices = get_available_gpus()[:EMILIA_WORKERS]
    if not available_devices:
        print("[!] No GPUs detected. Aborting.")
        return
        
    print(f"🚀 Starting {len(available_devices)} worker processes for {len(urls)} videos...")
    for device in available_devices:
        p = multiprocessing.Process(target=processing_worker, args=(url_queue, device))
        p.start()
        processes.append(p)

    for _ in processes:
        url_queue.put(None)

    url_queue.join()
    for p in processes:
        p.join()

    print("\n✅ Stage 2 Complete: All videos have been processed and uploaded to S3.")
    print(f"➡️  Intermediate results are in bucket: s3://{S3_BUCKET_NAME}/processed/")
    print("➡️  Next step: Run 'finalize_dataset.py' to create the Hugging Face dataset.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()

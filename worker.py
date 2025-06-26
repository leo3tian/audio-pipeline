import os
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
from yt_dlp import YoutubeDL
from hf_upload import upload_to_huggingface # We import our new upload function

# Config
# Path to the master list of URLs created by collect_urls.py.
# On a cloud platform, this would be a path to the file in a shared volume, e.g., '/mnt/s3/master_url_list.txt'.
URL_LIST_FILE = "master_url_list.txt" 
SAMPLE_RATE = 24000
EMILIA_WORKERS = 6 # Max number of GPUs to use
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"
# Hugging Face repo details for the upload.
HF_REPO_ID = "TO-DO"


def run_emilia_pipe(input_wav_file: str, output_dir: str, device: str):
    """
    Runs the Emilia-pipe on a specific audio file using a specific GPU.
    """
    print(f"GPU {device} - Processing file: {input_wav_file}")
    os.makedirs(output_dir, exist_ok=True)

    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline"
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)
    
    cmd = f"""
    source {conda_setup} && \
    conda activate {conda_env} && \
    export CUDA_VISIBLE_DEVICES={device} && \
    python {emilia_script} \
        --input_file_path '{input_wav_file}' \
        --config_path '{EMILIA_CONFIG_PATH}' \
        --output_dir '{output_dir}'
    """
    try:
        # Using DEVNULL for stdout to keep logs clean, stderr is piped for error checking.
        process = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"GPU {device} - ✅ Successfully processed: {input_wav_file}")
    except subprocess.CalledProcessError as e:
        # Decode stderr to print the actual error from the Emilia script
        error_message = e.stderr.decode()
        print(f"[!] Emilia error on GPU {device} for {input_wav_file}:\n---\n{error_message}\n---")
        raise

def processing_worker(url_queue: multiprocessing.JoinableQueue, device: str):
    """
    The main worker function. It runs in a continuous loop, pulling a URL from the queue
    and executing the full download-process-upload-cleanup pipeline.
    """
    while True:
        video_url = url_queue.get()
        if video_url is None:
            url_queue.task_done()
            break
        
        # Create a unique temporary directory for this specific video.
        # This prevents any file collisions between parallel workers.
        with tempfile.TemporaryDirectory(prefix="audiopipe_") as temp_dir:
            try:
                temp_path = Path(temp_dir)
                # The output from emilia will go into a 'processed' sub-folder inside the temp dir
                processed_dir = temp_path / "processed"

                # --- 1. DOWNLOAD ---
                print(f"GPU {device} - Downloading: {video_url}")
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
                    'postprocessor_args': ['-ar', str(SAMPLE_RATE)],
                    # Download directly into the temp directory
                    'outtmpl': str(temp_path / '%(id)s.%(ext)s'),
                    'quiet': True,
                    'ignoreerrors': True,
                }
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

                # Find the downloaded WAV file. Should only be one.
                downloaded_files = list(temp_path.glob("*.wav"))
                if not downloaded_files:
                    raise FileNotFoundError(f"yt-dlp failed to download the WAV file for URL: {video_url}")
                
                input_wav_path = str(downloaded_files[0])

                # --- 2. PROCESS ---
                run_emilia_pipe(input_wav_path, str(processed_dir), device)
                
                # --- 3. UPLOAD ---
                print(f"GPU {device} - Uploading results for: {video_url}")
                upload_to_huggingface(
                    local_processed_dir=str(processed_dir),
                    hf_repo_id=HF_REPO_ID,
                    video_id=downloaded_files[0].stem # Use video ID for subfolder on HF
                )
                
            except Exception as e:
                print(f"[!!!] CRITICAL FAILURE on GPU {device} for URL {video_url}: {e}")
                # The with-block handles cleanup of the temp directory automatically.
            finally:
                url_queue.task_done()

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        print("[!] nvidia-smi not found or failed. Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """
    Main orchestration function. Reads the master URL list, populates a queue,
    and starts a pool of worker processes, one for each available GPU.
    """
    try:
        with open(URL_LIST_FILE, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"[!] URL file '{URL_LIST_FILE}' is empty or not found.")
            return
    except FileNotFoundError:
        print(f"[!] Error: URL file not found at '{URL_LIST_FILE}'.")
        print("    Please run 'collect_urls.py' first.")
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

    # Add sentinel values (None) to the queue to tell workers to stop when done.
    for _ in processes:
        url_queue.put(None)

    # Wait for all URLs to be processed.
    url_queue.join()

    # Clean up worker processes.
    for p in processes:
        p.join()

    print("\n🎉🎉🎉 All videos have been processed and uploaded! 🎉🎉🎉")


if __name__ == "__main__":
    # Set start method to 'fork' for compatibility with CUDA in multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()

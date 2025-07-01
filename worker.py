import os
import json
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
import boto3
from streaming import StreamingDataset
import soundfile as sf
from itertools import islice # Import islice for manual sharding
import time
import tqdm

# --- Configuration ---
S3_INPUT_DIR = os.getenv("S3_INPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset")
S3_OUTPUT_DIR = os.getenv("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/processed")
EMILIA_WORKERS = 4
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"

def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix with cleaner logs."""
    s3_client = boto3.client("s3")
    files_to_upload = list(local_directory.rglob("*"))
    file_count = len(files_to_upload)
    if file_count == 0:
        print(f"    No files found in {local_directory} to upload.")
        return
    
    # This function is called from within a worker, so we avoid printing the large file list
    # print(f"    Uploading {file_count} files to s3://{s3_bucket}/{s3_prefix}...")
    
    for local_file in files_to_upload:
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)
            
    # print(f"    Finished uploading to s3://{s3_bucket}/{s3_prefix}")

def run_emilia_pipe(input_wav_file: str, output_dir: str, device: str):
    """
    Runs the Emilia-pipe on a specific audio file, capturing and streaming
    its output in real-time with a worker-specific prefix.
    """
    # This print is now handled by the main worker loop
    # print(f"GPU {device} - Processing file: {os.path.basename(input_wav_file)}")
    os.makedirs(output_dir, exist_ok=True)
    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline"
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)
    cmd = f"""
    source {conda_setup} && conda activate {conda_env} && export CUDA_VISIBLE_DEVICES={device} && \
    python {emilia_script} --input_file_path '{input_wav_file}' --config_path '{EMILIA_CONFIG_PATH}' --output_dir '{output_dir} --quiet'
    """

    process = subprocess.Popen(cmd, shell=True, executable="/bin/bash", 
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1)

    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(f"    [GPU {device}] {line.strip()}", flush=True)

    process.wait()
    if process.returncode != 0:
        print(f"[!] Emilia subprocess on GPU {device} failed for {input_wav_file}.")
        raise subprocess.CalledProcessError(process.returncode, cmd)

def processing_worker(rank: int, world_size: int, device: str, worker_cache_dir: str, progress_counter):
    """
    A worker process that uses a pre-sanitized index to read its slice of the dataset.
    """
    full_dataset = StreamingDataset(
        remote=S3_INPUT_DIR,
        local=worker_cache_dir,
        shuffle=False,
        batch_size=1
    )

    dataset_slice = islice(full_dataset, rank, None, world_size)

    for i, sample in enumerate(dataset_slice):
        global_index = rank + (i * world_size)
        video_id = sample['video_id']
        audio_bytes = sample['audio']
        
        print(f"GPU {device} (Rank {rank}) - Starting sample #{global_index}, video_id: {video_id}")

        with tempfile.TemporaryDirectory(prefix=f"worker_{device}_") as temp_dir:
            try:
                temp_path = Path(temp_dir)
                emilia_output_dir = temp_path / "processed"
                
                input_wav_path = temp_path / f"{video_id}.wav"
                with open(input_wav_path, 'wb') as f:
                    f.write(audio_bytes)

                run_emilia_pipe(str(input_wav_path), str(emilia_output_dir), device)
                
                s3_bucket_name = S3_OUTPUT_DIR.split('//')[1].split('/')[0]
                s3_prefix = f"processed/{video_id}"
                upload_directory_to_s3(emilia_output_dir, s3_bucket_name, s3_prefix)
                print(f"GPU {device} - ✅ Successfully processed and uploaded: {video_id}")

                # Increment the shared counter on success
                with progress_counter.get_lock():
                    progress_counter.value += 1
                
            except Exception as e:
                print(f"[!!!] CRITICAL FAILURE on GPU {device} for video_id {video_id}: {e}")

def progress_monitor(counter, total):
    """Monitors the shared counter and updates the tqdm progress bar."""
    with tqdm.tqdm(total=total, desc="Overall Progress") as pbar:
        last_value = 0
        while last_value < total:
            time.sleep(1) # Update every second
            current_value = counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
                last_value = current_value

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        print("[!] nvidia-smi not found. Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """Orchestrates the pool of GPU worker processes after sanitizing the dataset index."""
    available_devices = get_available_gpus()[:EMILIA_WORKERS]
    world_size = len(available_devices)

    if world_size == 0:
        print("[!] No GPUs detected. Aborting.")
        return
        
    shared_cache_dir = os.path.abspath("streaming_worker_cache")
    if os.path.exists(shared_cache_dir):
        shutil.rmtree(shared_cache_dir)
    os.makedirs(shared_cache_dir)

    print("Sanitizing remote dataset index...")
    try:
        s3_client = boto3.client("s3")
        s3_bucket, s3_prefix = S3_INPUT_DIR.replace("s3://", "").split('/', 1)
        
        original_index_key = os.path.join(s3_prefix, "index.json")
        sanitized_index_path = os.path.join(shared_cache_dir, "index.json")
        
        s3_client.download_file(s3_bucket, original_index_key, sanitized_index_path)

        with open(sanitized_index_path, 'r') as f:
            index_data = json.load(f)
        
        original_shard_count = len(index_data['shards'])
        sanitized_shards = [shard for shard in index_data['shards'] if shard.get('samples', 0) > 0]
        
        if len(sanitized_shards) < original_shard_count:
            print(f"  Found and removed {original_shard_count - len(sanitized_shards)} empty/invalid shards from the index.")

        index_data['shards'] = sanitized_shards
        
        # Calculate total samples from the sanitized index
        total_samples = sum(shard['samples'] for shard in sanitized_shards)

        with open(sanitized_index_path, 'w') as f:
            json.dump(index_data, f)
        
        print(f"  Sanitized index.json created locally. Total samples: {total_samples}")

    except Exception as e:
        print(f"[!!!] Could not download or sanitize index.json: {e}")
        print("[!!!] Aborting worker startup.")
        shutil.rmtree(shared_cache_dir, ignore_errors=True)
        return
    
    if total_samples == 0:
        print("[!] Dataset is empty after sanitization. Nothing to process.")
        shutil.rmtree(shared_cache_dir, ignore_errors=True)
        return

    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0)

    monitor_process = multiprocessing.Process(target=progress_monitor, args=(progress_counter, total_samples))
    monitor_process.start()

    print(f"🚀 Starting {world_size} worker processes to process {total_samples} videos...")
    processes = []
    try:
        for rank, device in enumerate(available_devices):
            worker_cache_dir = os.path.join(shared_cache_dir, f"worker_{rank}")
            os.makedirs(worker_cache_dir)
            shutil.copy(sanitized_index_path, os.path.join(worker_cache_dir, "index.json"))

            p = multiprocessing.Process(target=processing_worker, args=(rank, world_size, device, worker_cache_dir, progress_counter))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        # Wait for the monitor to finish. It will exit once the counter reaches the total.
        monitor_process.join()

    finally:
        if monitor_process.is_alive():
            monitor_process.terminate() # Forcefully stop the monitor if it's still running
        shutil.rmtree(shared_cache_dir, ignore_errors=True)

    print("\n✅ Stage 2 Complete: All videos have been processed and uploaded to S3.")
    print(f"➡️  Processed results are in bucket: {S3_OUTPUT_DIR}/")
    print("➡️  Next step: Run 'finalize_dataset.py' to create the Hugging Face dataset.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

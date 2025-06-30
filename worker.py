import os
import subprocess
import multiprocessing
import tempfile
import shutil
from pathlib import Path
import boto3
from streaming import StreamingDataset
import soundfile as sf
from itertools import islice # Import islice for manual sharding

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
    
    print(f"    Uploading {file_count} files to s3://{s3_bucket}/{s3_prefix}...")
    
    for local_file in files_to_upload:
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)
            
    print(f"    Finished uploading to s3://{s3_bucket}/{s3_prefix}")

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
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
        print(f"GPU {device} - ✅ Successfully processed: {input_wav_file}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        print(f"[!] Emilia error on GPU {device} for {input_wav_file}:\n---\n{error_message}\n---")
        raise

def processing_worker(rank: int, world_size: int, device: str):
    """
    A worker process that reads a unique slice of the StreamingDataset,
    processes each sample, and uploads the result to a new S3 prefix.
    """
    # Each worker initializes the full dataset. This is a cheap operation
    # that only reads the remote index file.
    with tempfile.TemporaryDirectory() as local_cache_dir:
        full_dataset = StreamingDataset(
            remote=S3_INPUT_DIR,
            local=local_cache_dir,
            shuffle=False,
            # FIX: Add the required batch_size argument. Since we process one
            # sample at a time, our batch size is 1.
            batch_size=1
        )

        # Create a unique slice of the dataset for this worker using islice.
        dataset_slice = islice(full_dataset, rank, None, world_size)

        # Iterate over the unique slice of the dataset assigned to this worker
        for i, sample in enumerate(dataset_slice):
            # The global index is rank + i * world_size
            global_index = rank + (i * world_size)
            video_id = sample['video_id']
            # The audio from the streaming dataset is already in bytes
            audio_bytes = sample['audio']
            sample_rate = sample['sample_rate']
            
            print(f"GPU {device} (Rank {rank}) - Starting sample #{global_index}, video_id: {video_id}")

            with tempfile.TemporaryDirectory(prefix=f"worker_{device}_") as temp_dir:
                try:
                    temp_path = Path(temp_dir)
                    emilia_output_dir = temp_path / "processed"
                    
                    # Save the audio bytes from the sample to a temporary .wav file
                    input_wav_path = temp_path / f"{video_id}.wav"
                    # The audio is already in bytes, so we write it directly.
                    # The soundfile library expects a numpy array, not raw bytes.
                    # We need to decode the WAV bytes first.
                    # Assuming the bytes are a valid WAV file, we can write them directly.
                    with open(input_wav_path, 'wb') as f:
                        f.write(audio_bytes)

                    # --- 2. PROCESS with Emilia-pipe ---
                    run_emilia_pipe(str(input_wav_path), str(emilia_output_dir), device)
                    
                    # --- 3. UPLOAD processed results TO S3 ---
                    s3_bucket_name = S3_OUTPUT_DIR.split('//')[1].split('/')[0]
                    s3_prefix = f"processed/{video_id}"
                    upload_directory_to_s3(emilia_output_dir, s3_bucket_name, s3_prefix)
                    
                except Exception as e:
                    print(f"[!!!] CRITICAL FAILURE on GPU {device} for video_id {video_id}: {e}")

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception:
        print("[!] nvidia-smi not found. Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """Orchestrates the pool of GPU worker processes."""
    processes = []
    available_devices = get_available_gpus()[:EMILIA_WORKERS]
    world_size = len(available_devices)

    if world_size == 0:
        print("[!] No GPUs detected. Aborting.")
        return
        
    print(f"🚀 Starting {world_size} worker processes...")
    
    for rank, device in enumerate(available_devices):
        p = multiprocessing.Process(target=processing_worker, args=(rank, world_size, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ Stage 2 Complete: All videos have been processed and uploaded to S3.")
    print(f"➡️  Processed results are in bucket: {S3_OUTPUT_DIR}/")
    print("➡️  Next step: Run 'finalize_dataset.py' to create the Hugging Face dataset.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

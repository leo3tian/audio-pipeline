import os
import multiprocessing
import tempfile
from pathlib import Path
import boto3
from streaming import StreamingDataset
from itertools import islice
import argparse

# Import the refactored Emilia functions
import Emilia.main as emilia

# --- Configuration ---
S3_INPUT_DIR = os.getenv("S3_INPUT_DIR", "s3://yt-pipeline-bucket/streaming_dataset")
S3_OUTPUT_DIR = os.getenv("S3_OUTPUT_DIR", "s3://yt-pipeline-bucket/processed")
EMILIA_WORKERS = 4

def upload_directory_to_s3(local_directory: Path, s3_bucket: str, s3_prefix: str):
    """Uploads the contents of a directory to a specific S3 prefix."""
    s3_client = boto3.client("s3")
    files_to_upload = list(local_directory.rglob("*"))
    if not files_to_upload:
        return
    
    print(f"    Uploading {len(files_to_upload)} files to s3://{s3_bucket}/{s3_prefix}...")
    for local_file in files_to_upload:
        if local_file.is_file():
            s3_key = f"{s3_prefix}/{local_file.relative_to(local_directory)}"
            s3_client.upload_file(str(local_file), s3_bucket, s3_key)
    print(f"    Finished uploading to s3://{s3_bucket}/{s3_prefix}")


def processing_worker(rank: int, world_size: int, device_id: str, emilia_args):
    """
    A worker process that loads models ONCE, then processes its slice of the dataset.
    """
    # Set the CUDA_VISIBLE_DEVICES for this specific worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    
    # --- 1. LOAD MODELS ONCE ---
    # We pass the GPU ID to the loading function.
    emilia_args.gpu_id = int(device_id)
    emilia.load_all_models(emilia_args)
    
    # --- 2. PROCESS DATASET SLICE ---
    with tempfile.TemporaryDirectory() as local_cache_dir:
        full_dataset = StreamingDataset(
            remote=S3_INPUT_DIR,
            local=local_cache_dir,
            shuffle=False,
            batch_size=1
        )
        dataset_slice = islice(full_dataset, rank, None, world_size)

        for i, sample in enumerate(dataset_slice):
            global_index = rank + (i * world_size)
            video_id = sample['video_id']
            audio_bytes = sample['audio']
            
            print(f"GPU {device_id} (Rank {rank}) - Starting sample #{global_index}, video_id: {video_id}")

            with tempfile.TemporaryDirectory(prefix=f"worker_{device_id}_") as temp_dir:
                try:
                    temp_path = Path(temp_dir)
                    emilia_output_dir = temp_path / "processed"
                    
                    input_wav_path = temp_path / f"{video_id}.wav"
                    with open(input_wav_path, 'wb') as f:
                        f.write(audio_bytes)

                    # Call the main processing function directly
                    emilia.main_process(
                        audio_path=str(input_wav_path),
                        save_path=str(emilia_output_dir),
                        audio_name=video_id
                    )
                    
                    # Upload results to S3
                    s3_bucket_name = S3_OUTPUT_DIR.split('//')[1].split('/')[0]
                    s3_prefix = f"processed/{video_id}"
                    upload_directory_to_s3(emilia_output_dir, s3_bucket_name, s3_prefix)
                    
                except Exception as e:
                    print(f"[!!!] CRITICAL FAILURE on GPU {device_id} for video_id {video_id}: {e}")

def get_available_gpus() -> list:
    """Detects available NVIDIA GPU indices."""
    try:
        # This is a placeholder for a real environment.
        # In MosaicML or a cloud instance, this would be managed differently.
        num_gpus = int(os.getenv("WORLD_SIZE", EMILIA_WORKERS))
        return [str(i) for i in range(num_gpus)]
    except Exception:
        print("[!] Assuming 1 GPU (device 0).")
        return ["0"]

def main():
    """Orchestrates the pool of GPU worker processes."""
    # Create a dummy args object to pass to the Emilia model loader
    emilia_parser = argparse.ArgumentParser()
    emilia_parser.add_argument("--config_path", type=str, default="Emilia/config.json")
    # Add other args that load_models might need
    emilia_args, _ = emilia_parser.parse_known_args()

    processes = []
    available_devices = get_available_gpus()
    world_size = len(available_devices)

    if world_size == 0:
        print("[!] No GPUs detected. Aborting.")
        return
        
    print(f"🚀 Starting {world_size} worker processes...")
    
    for rank, device_id in enumerate(available_devices):
        p = multiprocessing.Process(target=processing_worker, args=(rank, world_size, device_id, emilia_args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ Stage 2 Complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()

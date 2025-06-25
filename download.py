import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import JoinableQueue
from pathlib import Path
from yt_dlp import YoutubeDL
from typing import List
import threading

# config
CHANNEL_URLS = [
    "https://www.youtube.com/@CodeAesthetic"
]
DOWNLOAD_DIR = "downloads"
PROCESSED_DIR = "processed"
SAMPLE_RATE = 24000
YT_MAX_WORKERS = 2
EMILIA_WORKERS = 6
EMILIA_PIPE_PATH = "Emilia/main.py"
EMILIA_CONFIG_PATH = "Emilia/config.json"

# downloads all videos from youtube channel
def download_audio_from_channel(channel_url: str, queue: JoinableQueue):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'postprocessor_args': ['-ar', str(SAMPLE_RATE)],
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(uploader_id)s/%(id)s/%(id)s.%(ext)s'),
        'quiet': True,
        'ignoreerrors': True,
        # 'extractor_args': {'youtube': ['player_client=web']},
    }

    try:
        flat_opts = ydl_opts.copy()
        print(channel_url + " - 🔗 Extracting video URLs")
        flat_opts['extract_flat'] = True
        with YoutubeDL(flat_opts) as ydl_flat:
            info = ydl_flat.extract_info(channel_url, download=False)
            if not info or not info.get("entries"):
                return
            video_urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info['entries'] if e and 'id' in e]

        print(channel_url + f" - 🔄 Downloading and enqueueing {len(video_urls)} audio files")
        full_opts = ydl_opts.copy()
        full_opts['extract_flat'] = False
        for video_url in video_urls:
            with YoutubeDL(full_opts) as ydl_full:
                ydl_full.download([video_url])
            wav_paths = list(Path(DOWNLOAD_DIR).rglob(f"{video_url.split('=')[-1]}.wav"))
            for path in wav_paths:
                queue.put(str(path.resolve()))
        print(channel_url + " - ✅ Download and enqueue complete")
    except Exception as e:
        print(f"[!] Error downloading from {channel_url}: {e}")


def download_and_enqueue(queue: JoinableQueue):
    for url in CHANNEL_URLS:
        download_audio_from_channel(url, queue)


# runs emilia-pipe with a given GPU (passed thru device) to run on
# now includes output folder path argument to override default save location in Emilia
def run_emilia_pipe(input_wav: str, output_dir: str, device: str):
    print("GPU " + device + " - Processing " + input_wav)
    os.makedirs(output_dir, exist_ok=True)

    conda_setup = "/opt/conda/etc/profile.d/conda.sh"
    conda_env = "AudioPipeline"
    emilia_script = os.path.abspath(EMILIA_PIPE_PATH)

    # Set memory cap via env var and Python torch config
    # Force device visibility + log memory cap in subprocess
    cmd = f"""
    source {conda_setup} && \
    conda activate {conda_env} && \
    export TORCH_HOME="/tmp/torch_cache_{device}" && \
    python -c "import torch; torch.cuda.set_per_process_memory_fraction(0.8, 0)" && \
    python {emilia_script} --input_folder_path '{input_wav}' --config_path '{EMILIA_CONFIG_PATH}' --output_dir '{output_dir}'
    """
    try:
        subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"[!] Emilia error ({input_wav}):\n{e.stderr.decode()}")


# === WORKER ===
def processing_worker(queue: JoinableQueue, device: str):
    while True:
        wav_path = queue.get()
        if wav_path is None:
            queue.task_done()
            break
        wav_path = Path(wav_path)
        try:
            channel = wav_path.parts[-3]
            video_id = wav_path.stem
            outdir = os.path.join(PROCESSED_DIR, channel, video_id)
            run_emilia_pipe(str(wav_path.parent), outdir, device)
        except Exception as e:
            print(f"[!] Failed to process {wav_path}: {e}")
        finally:
            queue.task_done()


def get_available_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [line.strip() for line in output.decode("utf-8").splitlines()]
    except Exception as e:
        print(f"[!] Failed to detect GPUs: {e}")
        return []


# === MAIN ===
def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    queue = JoinableQueue()
    processes = []

    # Start processing workers
    available_devices = get_available_gpus()[:EMILIA_WORKERS]
    for device in available_devices:
        p = multiprocessing.Process(target=processing_worker, args=(queue, device))
        p.start()
        processes.append(p)

    # Start download + enqueue in a thread
    downloader_thread = threading.Thread(target=download_and_enqueue, args=(queue,))
    downloader_thread.start()
    downloader_thread.join()

    for _ in processes:
        queue.put(None)

    queue.join()
    for p in processes:
        p.join()

    print("✅ All audio processed with Emilia-Pipe.")


if __name__ == "__main__":
    main()
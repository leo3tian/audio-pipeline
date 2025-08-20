## Overview
This repo builds an audio dataset in three steps:

1) Download to object storage (S3/R2) — `src/downloading/` (this is legacy, see podcastindex-dl repo)
2) Process on GPUs with Emilia Pipe — `src/processing/`
3) Upload batches to Hugging Face — `src/uploading/`

Setup instructions for step 2 are also included below. 

## 1) Downloading (legacy)
Folder: `src/downloading/`

What it does:
- `get_urls.py` reads YouTube channels/playlists and pushes tasks to SQS (`AWS_QUEUE_DOWNLOAD`).
- `download_worker.py` consumes tasks, download audio with `yt-dlp`, convert to M4A, and upload to S3 under `raw_audio/`.

Notes:
- These downloading scripts are legacy at this point - we've pivoted to downloading podcasts via podcastindex-dataset (see podcastindex-dl repo)

Run (example):
```bash
# 1) Seed tasks (edit CHANNEL_URLS in the script first)
python src/downloading/get_urls.py

# 2) Downloaders
python src/downloading/download_worker.py
```

Env (examples):
```bash
export AWS_QUEUE_DOWNLOAD="https://sqs.<region>.amazonaws.com/<acct>/<queue>"
export AWS_REGION=us-east-2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Notes: downloading targets AWS S3 for raw audio. Please follow YouTube terms/copyright.

## 2) Processing on GPUs (Emilia Pipe)
Folder: `src/processing/`

What it does:
- `setup_processing_tasks.py` scans `raw_audio/` and writes `.task` files under `tasks/processing_todo/` in R2/S3.
- `gpu_worker.py` claims a task, downloads one episode, runs Emilia (separation, diarization, VAD, ASR, DNSMOS), and uploads results to `processed/<language>/<episode_id>/`.
- Launches multiple worker processes across detected GPUs.

Notes:
- This is the main script in the repo; its why we are using a docker image. It's designed to run on 8x GPU nodes and run Emilia-Pipe in parallel across GPUs. 
- See how to build the docker image below.

## Setup Instructions

Download sig_bak_ovr.onnx and UVR-MDX-NET-Inst_HQ_3.onnx into Emilia/models and update their path (if needed) in Emilia/config

Build the image:
```bash
docker build -t emilia-pipeline .
```

(Optional) push:
```bash
# Docker Hub example
docker tag emilia-pipeline youruser/emilia-pipeline:latest
docker push youruser/emilia-pipeline:latest
```

Key env (can also set keys in the kubernetes env variables & secrets):
```bash
# R2 (S3-compatible)
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=<bucket>

# Prefixes
export RAW_AUDIO_PREFIX=raw_audio/
export PROCESSED_PREFIX=processed/
export TASKS_BASE_PREFIX=tasks/

# Models
export HF_TOKEN=<huggingface_token>
# Worker tuning
export WORKERS_PER_GPU=1
export MAX_GPUS=8
```

**Kubernetes is the preferred way to run.** Check out /examples/:
- `emiliapipe.yaml` starts a kubernetes deployment that runs the GPU processing script (designed for H100s but tunable)
- `autokicker.yaml` is a kubernetes deployment for monitors the emiliapipe deployment and letting other jobs run first, before . It was made to kick off emiliapipe pods when other pods need to run, so other team members can use GPUs without being blocked by the emiliapipe jobs. This is working as intended when:
    - There are no emiliapipe pods when other pods are pending
    - Otherwise, there should always be any amount of emiliapipe pods running and exactly one emiliapipe pod pending (this pending pod is how the script knows when to stop adding emiliapipe pods)

Run on a GPU node (may be deprecated):
```bash
# .env should contain the envs above
docker run --rm -it \
  --gpus all \
  --env-file .env \
  -v $(pwd):/workspace \
  -w /workspace \
  emilia-pipeline \
  bash -lc "python src/processing/setup_processing_tasks.py && python src/processing/gpu_worker.py"
```

Notes:
- `Emilia/` is at repo root and is auto-resolved; override with `EMILIA_PATH` if needed.
- Control parallelism with `WORKERS_PER_GPU` and `MAX_GPUS`.

## Changes to Emilia Pipe (high‑level)
- Long‑running workers instead of one‑shot scripts: models are loaded once and many episodes are processed per node. Orchestration via `src/processing/gpu_worker.py` replaces the vendor `Emilia/main_multi.py` for our use case.
- Performance and stability knobs: added environment variables to control memory/throughput (e.g., `SEPARATION_CHUNKS`, ORT CUDA limits), chunked source separation with overlap, soxr‑based resampling (44.1k/16k where needed), and multi‑threaded MP3 export; each worker pins to a specific GPU device.
- DNSMOS filtering removed from outputs: we still compute DNSMOS for analysis, but we don’t drop segments by score; uploads use `all_segments.json` so the research team receives the full dataset.

## 3) Uploading to Hugging Face
Folder: `src/uploading/`

What it does:
- `setup_upload_tasks.py` scans `processed/`, creates batch definitions in `tasks/upload_batches/`, and enqueues SQS messages.
- `upload_worker.py` downloads files, writes JSON sidecars, tars content, and uploads to your HF dataset repo.

Env (examples):
```bash
export S3_BUCKET_NAME=<bucket-with-processed-data>
export AWS_REGION=us-east-2
export SQS_QUEUE_URL=https://sqs.<region>.amazonaws.com/<acct>/<queue>
export HF_REPO_ID=<org-or-user>/<dataset>
export HF_TOKEN=<huggingface_token>
```

Run:
```bash
python src/uploading/setup_upload_tasks.py
python src/uploading/upload_worker.py
```

Notes:
- A legacy single-process path is available at `src/uploading/LEGACY_upload_dataset.py` if you want to run end-to-end without SQS workers.

## Development
- Python deps are in `requirements.txt` (used by the Dockerfile).
- `cache_models.py` pre-caches models during the image build.

## Credits
- Emilia Pipe inspiration and related ideas draw from OpenMMLab Amphion and community work.


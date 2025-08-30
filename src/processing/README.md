# Audio Processing

Two scripts to process raw audio from R2 and upload outputs back to R2.

### File Breakdown

`setup_processing_tasks.py` scans the `raw_audio` prefix in R2 (via the RAW_AUDIO_PREFIX environment variable) and creates task files under a processing todo prefix.

`gpu_worker.py` consumes task files from R2, processes audio with the Emilia pipeline, and uploads results to R2.

## How To Run

High level steps:
1. Download models
2. Build docker image
3. Run on kubernetes

### 1. Download models

Download sig_bak_ovr.onnx and UVR-MDX-NET-Inst_HQ_3.onnx into Emilia/models and update their path (if needed) in Emilia/config

### 2. Build docker image

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

### 3. Run on Kubernetes

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
export WORKERS_PER_GPU=2
export MAX_GPUS=8
```

For kubernetes config files, check out /examples/. I run these by doing `kubectl apply -f emiliapipe.yaml`:
- `emiliapipe.yaml` starts a kubernetes deployment that runs the GPU processing script (designed for H100s but tunable)
- `autokicker.yaml` is a kubernetes deployment for monitors the emiliapipe deployment and letting other jobs run first, before . It was made to kick off emiliapipe pods when other pods need to run, so other team members can use GPUs without being blocked by the emiliapipe jobs. This is working as intended when:
    - There are no emiliapipe pods when other pods are pending
    - Otherwise, there should always be any amount of emiliapipe pods running and exactly one emiliapipe pod pending (this pending pod is how the script knows when to stop adding emiliapipe pods)

**Kubernetes is the preferred way to run, but other methods work as well. You simply have to pull the docker image, set up env variables, then run gpu_worker.py like this (may be deprecated):**

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

## Notes
This codebase contains Emilia-Pipe, but with a few key changes:
- Long‑running workers instead of one‑shot scripts: models are loaded once and many episodes are processed per node. Orchestration via `src/processing/gpu_worker.py` replaces the vendor `Emilia/main_multi.py` for our use case.
- Performance and stability knobs: added environment variables to control memory/throughput (e.g., `SEPARATION_CHUNKS`, ORT CUDA limits), chunked source separation with overlap, soxr‑based resampling (44.1k/16k where needed), and multi‑threaded MP3 export; each worker pins to a specific GPU device.
- DNSMOS filtering removed from outputs: we still compute DNSMOS for analysis, but we don’t drop segments by score; uploads use `all_segments.json` so the research team receives the full dataset.
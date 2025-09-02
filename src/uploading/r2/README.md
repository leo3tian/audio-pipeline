# Dataset Builder

Two-mode script to build and upload a Hugging Face dataset from R2-processed audio.

### File Breakdown

`build_dataset.py` scans `processed/<language>/<episode_id>/` prefixes in R2 and either:

- Scan mode (`--scan`): appends newly discovered episode prefixes to `work_plan.json` (append-only; preserves existing order, avoids duplicates).
- Process mode: streams episode data, writes Parquet shard(s), and appends them to a Hugging Face dataset repo. Progress is tracked in `progress.log` so runs can resume safely.

## How To Run

High level steps:
1. Set environment variables
2. (Optional) Scan and create a work plan
3. Process and upload chunks

### 1. Set environment variables

Required credentials and configuration:
```bash
# R2 (S3-compatible)
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=<bucket>

# Hugging Face
export HF_REPO_ID=<org-or-user/your-dataset>
export HF_TOKEN=<huggingface_token>

# Performance tuning (optional)
export EPISODES_PER_CHUNK=1000           # episodes per processing chunk
export DOWNLOAD_WORKERS=128              # parallel episode fetch threads
export PARQUET_MAX_BYTES=$((2*1024*1024*1024))      # ~2 GiB max per parquet
export PARQUET_ROWGROUP_BYTES=$((128*1024*1024))    # ~128 MiB per row group

# Enable HF transfer acceleration
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Files `work_plan.json` and `progress.log` are written to the current working directory (where you run the script). Run all cooperating processes from the same directory if you want them to share plan/progress.

### 2. Scan and create a work plan

Scans R2 for available languages and episode prefixes and appends them to `work_plan.json`:
```bash
python src/uploading/r2/build_dataset.py --scan \
  --language en \
  --language de
```
Notes:
- You can repeat `--language` to include multiple languages. If omitted, all languages found under `processed/` are included.
- Running `--scan` again will merge in only new episode prefixes; it will not overwrite existing entries.

### 3. Process and upload chunks

Build Parquet shard(s) and append to the HF dataset repo. Progress is logged to `progress.log` and the run is safe to resume.
```bash
python src/uploading/r2/build_dataset.py
```

Distribute work across multiple machines using modulo selection:
```bash
# Machine 0 of 4
python src/uploading/r2/build_dataset.py --chunk-mod 4 --chunk-rem 0

# Machine 1 of 4
python src/uploading/r2/build_dataset.py --chunk-mod 4 --chunk-rem 1
```

If you run multiple processes, start them in the same directory to share `work_plan.json` and `progress.log`. Balance total concurrency by lowering `DOWNLOAD_WORKERS` per process.

## Notes

- Data schema per row: `audio` (struct: `path`, `bytes`), `text`, `speaker_id`, `duration_seconds`, `dnsmos`, `language`.
- We perform language normalization (because the PodcastIndex dataset contains accents such as en-ZH): e.g., "en-US"/"en_gb"/"ENG" → `en`; unknowns mapped to `unknown`.
- Dependencies: `datasets`, `huggingface_hub[hf_transfer]`, `pyarrow`, `boto3`, `python-dotenv`, `tqdm`.

### Shard layout

- Parquet shards are stored under `data/<language>/` with names like:
  - `chunk-00012.parquet` (single-part), or
  - `chunk-00012-part-00.parquet`, `chunk-00012-part-01.parquet` (size-partitioned)
  Uploads use append-only commits so earlier shards are not pruned.


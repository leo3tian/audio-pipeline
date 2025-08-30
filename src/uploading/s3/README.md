# S3 to Hugging Face Uploader

Scripts to upload processed audio from an S3 bucket to a Hugging Face dataset. This folder contains two distinct systems: a robust, distributed uploader for very large datasets and a simpler, single-file legacy uploader for smaller jobs.

## Distributed Uploader (Recommended for Large Datasets)

A scalable, two-stage system for uploading terabytes of data. It uses S3 for task definitions and SQS for queueing, allowing many workers to process uploads in parallel.

### File Breakdown

- `setup_upload_tasks.py`: Scans the `processed/` prefix in S3, creates batch definition files (`.jsonl`), uploads them to S3, and enqueues a task message in SQS for each batch. **Run this once.**
- `upload_worker.py`: A worker script that polls the SQS queue for tasks. When a task is received, it downloads the corresponding batch definition from S3, fetches the audio, creates a `.tar` archive, and uploads it to Hugging Face. **Run multiple instances of this.**

### How to Run

High level steps:
1. Set environment variables.
2. Run `setup_upload_tasks.py` to scan S3 and populate the queue.
3. Run one or more `upload_worker.py` instances to process the queue.

#### 1. Set environment variables
```bash
# AWS credentials should be configured as well (e.g., via ~/.aws/credentials)
export S3_BUCKET_NAME=<your-s3-bucket>
export SQS_QUEUE_URL=<your-sqs-queue-url>
export AWS_REGION=<your-aws-region> # e.g., us-east-1

# Hugging Face
export HF_REPO_ID=<org-or-user/your-dataset>
export HF_TOKEN=<huggingface_token>
```

#### 2. Run the Task Setup Script
This script inventories all processed files in S3 and creates the upload tasks.
```bash
python src/uploading/s3/setup_upload_tasks.py
```

#### 3. Run the Upload Workers
You can run many workers on different machines to maximize upload speed.
```bash
python src/uploading/s3/upload_worker.py
```

---

## Legacy Uploader (For Smaller Datasets)

A single script to upload files from S3 to Hugging Face. It runs on a single machine and is suitable for smaller-scale uploads where a distributed setup is not necessary.

### File Breakdown

- `LEGACY_upload_dataset.py`: A monolithic script that scans S3, downloads files in batches, creates local `.tar` archives, and uploads them directly to Hugging Face.

### How to Run

#### 1. Set environment variables
```bash
# AWS credentials should be configured
export S3_BUCKET_NAME=<your-s3-bucket>

# Hugging Face
export HF_REPO_ID=<org-or-user/your-dataset>
export HF_TOKEN=<huggingface_token>
```

#### 2. Run the Script
```bash
python src/uploading/s3/LEGACY_upload_dataset.py
```

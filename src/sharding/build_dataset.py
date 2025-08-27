import os
import json
import gzip
import tempfile
import shutil
import argparse
import concurrent.futures
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import uuid
import random
from urllib.parse import quote


import pyarrow as pa
import pyarrow.parquet as pq


from huggingface_hub import HfApi, CommitOperationAdd
load_dotenv()

# --- 1. Configuration ---
# pip install datasets "huggingface_hub[hf_transfer]" boto3 pyarrow
# Load from environment variables for security and flexibility
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

HF_REPO_ID = os.environ.get("HF_REPO_ID") # e.g., "your-org/your-50TB-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Batching and Concurrency Controls ---
# Keep chunks modest; we stream to Parquet shards, not memory.
EPISODES_PER_CHUNK = int(os.environ.get("EPISODES_PER_CHUNK", "20000"))
# Number of parallel threads to fetch metadata from R2.
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "64"))

# Parquet shard policy (override via env or CLI flags)
MAX_ROWS_PER_SHARD = int(os.environ.get("MAX_ROWS_PER_SHARD", "250000"))
WRITE_BATCH_ROWS = int(os.environ.get("WRITE_BATCH_ROWS", "5000"))


# --- File Paths for State Management ---
PROGRESS_LOG = "src/sharding/progress.log"
CHUNK_STATUS_DIR = "src/sharding/chunk_status"
WORK_PLAN_FILE = "src/sharding/work_plan.json.gz"


# --- 2. R2 Client and Helper Functions ---

def get_r2_client():
    """Initializes and returns the Boto3 client for R2 with pooling and retries."""
    return boto3.client(
        service_name="s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=256,
            read_timeout=60,
            connect_timeout=20,
        ),
    )

# Single shared client across threads
R2_CLIENT = get_r2_client()

HF_API = HfApi()

# Fixed Parquet schema to avoid drift
SCHEMA = pa.schema([
    ("audio_url", pa.string()),
    ("episode_id", pa.string()),
    ("segment_index", pa.int32()),
    ("text", pa.string()),
    ("speaker_id", pa.string()),
    ("duration_seconds", pa.float32()),
    ("dnsmos", pa.float32()),
    ("language", pa.string()),
])

# Tiny in-process cache of paths committed during this run (avoid duplicate checks)
_JUST_COMMITTED_PATHS: set = set()

def _normalize_language_code(lang_str):
    """
    Normalizes a messy language string from a folder name into a
    standard base language code.
    e.g., "en-US", "en_gb", "EN", "eng" -> "en"
    """
    if not lang_str:
        return "unknown"
    
    code = lang_str.lower().strip()
    
    # Handle special mappings for non-standard or 3-letter codes
    if code == 'deu': return 'de'
    if code == 'eng': return 'en'
    if code == 'srp': return 'sr'
    if code == 'in': return 'id' # Old code for Indonesian
    if code in ['unite', 'un', 'unknown', '']: return 'unknown'
    
    # General rule: take the primary subtag before a hyphen or underscore
    base_code = code.split('-')[0].split('_')[0]
    return base_code

def list_all_episode_prefixes_by_language(r2_client, limit_per_lang: Optional[int] = None, max_languages: Optional[int] = None):
    """
    Efficiently lists unique episode prefixes from R2 grouped by normalized base language.
    Parallelizes episode listing per language to speed up scanning.
    """
    t0 = time.time()
    print("[scan] Listing all episode prefixes from R2 by language (parallel)...")
    prefixes_by_lang = {}
    paginator = r2_client.get_paginator('list_objects_v2')

    # 1) List language folders (e.g., 'processed/en/')
    print("[scan] Fetching language prefixes under 'processed/' ...")
    lang_pages = paginator.paginate(Bucket=R2_BUCKET, Prefix="processed/", Delimiter='/')
    language_entries = []  # [(normalized_language, lang_prefix)]
    lang_page_count = 0
    for page in lang_pages:
        lang_page_count += 1
        cps = page.get('CommonPrefixes', [])
        print(f"[scan] Language page {lang_page_count}: {len(cps)} prefixes")
        for lang_prefix_data in cps:
            lang_prefix = lang_prefix_data.get('Prefix')
            raw_language = lang_prefix.strip('/').split('/')[-1]
            normalized_language = _normalize_language_code(raw_language)
            language_entries.append((normalized_language, lang_prefix))
    print(f"[scan] Total languages discovered: {len(language_entries)}")

    # Optionally cap number of languages for testing
    if max_languages is not None:
        language_entries = language_entries[:max_languages]
        print(f"[scan] Max languages cap applied: {len(language_entries)}")

    # 2) Scan each language in parallel
    import concurrent.futures

    def scan_one_language(entry):
        normalized_language, lang_prefix = entry
        start = time.time()
        print(f"[scan] -> Scan start lang='{normalized_language}' prefix='{lang_prefix}'")
        collected = []
        try:
            ep_pages = r2_client.get_paginator('list_objects_v2').paginate(
                Bucket=R2_BUCKET, Prefix=lang_prefix, Delimiter='/'
            )
            ep_page_count = 0
            for ep_page in ep_pages:
                ep_page_count += 1
                eps = ep_page.get('CommonPrefixes', [])
                print(f"[scan:{normalized_language}] episode page {ep_page_count}: {len(eps)} prefixes")
                for ep_prefix in eps:
                    collected.append(ep_prefix.get('Prefix'))
                    if limit_per_lang and len(collected) >= limit_per_lang:
                        dur = time.time() - start
                        print(f"[scan:{normalized_language}] limit {limit_per_lang} reached in {dur:.2f}s")
                        return normalized_language, collected
        except Exception as e:
            print(f"[scan:{normalized_language}] Warning: failed scanning '{lang_prefix}': {e}")
        dur = time.time() - start
        print(f"[scan] <- Scan end lang='{normalized_language}', found {len(collected)} episodes in {dur:.2f}s")
        return normalized_language, collected

    print(f"[scan] Starting parallel episode listing across {len(language_entries)} languages ...")
    t_lang = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for idx, (normalized_language, collected) in enumerate(executor.map(scan_one_language, language_entries), start=1):
            if idx % 5 == 0:
                print(f"[scan] progress: {idx}/{len(language_entries)} languages scanned")
            if not collected:
                continue
            existing = prefixes_by_lang.setdefault(normalized_language, [])
            existing.extend(collected)
    print(f"[scan] Parallel episode listing complete in {time.time() - t_lang:.2f}s")

    for lang, prefixes in prefixes_by_lang.items():
        print(f"[scan]   Found {len(prefixes)} episodes for language: '{lang}'")

    print(f"[scan] Total scan duration: {time.time() - t0:.2f}s")
    return prefixes_by_lang


def load_completed_chunks():
    """Loads the set of chunk IDs that have already been processed (e.g., 'en-0', 'es-1')."""
    if not os.path.exists(PROGRESS_LOG):
        return set()
    with open(PROGRESS_LOG, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def mark_chunk_as_completed(chunk_id):
    """Appends a completed chunk ID to the progress log."""
    with open(PROGRESS_LOG, 'a') as f:
        f.write(f"{chunk_id}\n")


# --- 3. Data Fetching (metadata only) ---

def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_audio_url(key: str) -> str:
    """Construct a URL using the configured R2 endpoint and bucket (keys URL-encoded)."""
    endpoint = (R2_ENDPOINT_URL or '').rstrip('/')
    path = quote(key, safe="/")
    return f"{endpoint}/{R2_BUCKET}/{path}"


def fetch_episode_manifest_rows(prefix: str) -> List[Dict[str, Any]]:
    """Fetches metadata for one episode and returns manifest rows without audio bytes."""
    rows: List[Dict[str, Any]] = []
    try:
        metadata_key = f"{prefix}all_segments.json"
        obj = R2_CLIENT.get_object(Bucket=R2_BUCKET, Key=metadata_key)
        segments = json.loads(obj['Body'].read())

        episode_id = prefix.strip('/').split('/')[-1]
        for i, segment in enumerate(segments):
            audio_filename = f"{episode_id}_{i:06d}.mp3"
            audio_key = f"{prefix}{audio_filename}"
            rows.append({
                "audio_url": _build_audio_url(audio_key),
                "episode_id": episode_id,
                "segment_index": int(i),
                "text": segment.get("text", ""),
                "speaker_id": str(segment.get("speaker", "UNKNOWN")),
                "duration_seconds": _to_float(segment.get("end", 0.0)) - _to_float(segment.get("start", 0.0)),
                "dnsmos": _to_float(segment.get("dnsmos", 0.0), 0.0),
                "language": str(segment.get("language", "UNKNOWN")),
            })
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            print(f"[fetch] Missing file under prefix {prefix}: {e}")
            return []
        print(f"[fetch] Boto3 ClientError on prefix {prefix}: {e}")
        return []
    except Exception as e:
        print(f"[fetch] Error processing prefix {prefix}: {e}")
        return []
    return rows


class RollingParquetWriter:
    """Stream rows into Parquet with automatic shard rotation by max rows."""

    def __init__(self, language: str, current_chunk: str, max_rows_per_shard: int, write_batch_rows: int) -> None:
        self.language = language
        self.current_chunk = current_chunk
        self.max_rows_per_shard = max_rows_per_shard
        self.write_batch_rows = write_batch_rows
        self.tmp_dir = tempfile.mkdtemp(prefix=f"parquet_{language}_")
        self.current_writer: Optional[pq.ParquetWriter] = None
        self.current_rows = 0
        self.current_file_path: Optional[str] = None
        self.shard_index = 0

    def _new_file_path(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        uid = uuid.uuid4().hex[:8]
        fname = f"{self.language}-{self.current_chunk}-shard-{ts}-{self.shard_index:05d}-{uid}.parquet"
        return os.path.join(self.tmp_dir, fname)

    def _ensure_writer(self, example_batch: List[Dict[str, Any]]) -> None:
        if self.current_writer is not None:
            return
        table = pa.Table.from_pylist(example_batch, schema=SCHEMA)
        self.current_file_path = self._new_file_path()
        self.current_writer = pq.ParquetWriter(
            self.current_file_path,
            SCHEMA,
            compression="zstd",
            write_statistics=True,
        )
        # Immediately write the first batch
        self.current_writer.write_table(table)
        self.current_rows = len(example_batch)

    def write_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Write rows; rotate shard if threshold reached. Returns list of committed local file paths."""
        if not rows:
            return []
        if self.current_writer is None:
            # Start writer with an initial batch up to write_batch_rows
            initial = rows[: self.write_batch_rows]
            self._ensure_writer(initial)
            remaining = rows[self.write_batch_rows :]
        else:
            remaining = rows

        committed_paths: List[str] = []

        # Write remaining rows in batches
        idx = 0
        while idx < len(remaining):
            batch = remaining[idx : idx + self.write_batch_rows]
            table = pa.Table.from_pylist(batch, schema=SCHEMA)
            self.current_writer.write_table(table, row_group_size=self.write_batch_rows)
            self.current_rows += len(batch)
            idx += len(batch)
            if self.current_rows >= self.max_rows_per_shard:
                # rotate
                p = self.close_and_rotate()
                if p:
                    committed_paths.append(p)
        return committed_paths

    def close_and_rotate(self) -> Optional[str]:
        if self.current_writer is None:
            return None
        self.current_writer.close()
        local_path = self.current_file_path
        self.current_writer = None
        self.current_rows = 0
        self.current_file_path = None
        self.shard_index += 1
        return local_path

    def finalize(self) -> Optional[str]:
        return self.close_and_rotate()

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _repo_path_exists(path_in_repo: str) -> bool:
    try:
        HF_API.repo_file_info(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            token=HF_TOKEN,
        )
        return True
    except Exception:
        return False


def _commit_parquet_to_hub(local_path: str, language: str, shard_basename: Optional[str] = None) -> bool:
    """Append a Parquet shard to the dataset repo using low-level commit API."""
    if not os.path.exists(local_path):
        return False
    base_name = shard_basename or os.path.basename(local_path)
    repo_path = f"data/{language}/{base_name}"
    if repo_path in _JUST_COMMITTED_PATHS or _repo_path_exists(repo_path):
        print(f"[commit] SKIP exists {repo_path}")
        return True

    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            HF_API.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                operations=[
                    CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path),
                ],
                commit_message=f"Add shard {base_name} for lang '{language}'",
            )
            _JUST_COMMITTED_PATHS.add(repo_path)
            return True
        except Exception as e:
            attempts += 1
            if attempts >= max_attempts:
                print(f"[commit] ERROR committing {local_path} -> {repo_path}: {e}")
                return False
            backoff = min(60, (2 ** attempts)) + random.random()
            print(f"[commit] retry {attempts}/{max_attempts} in {backoff:.1f}s: {e}")
            time.sleep(backoff)


def _commit_chunk_manifest(language: str, chunk_id: str, rows: int, shards: int) -> None:
    """Commit a tiny manifest JSON per chunk for cross-process visibility."""
    try:
        payload = json.dumps({
            "chunk": chunk_id,
            "language": language,
            "rows": rows,
            "shards": shards,
            "ts": datetime.utcnow().isoformat() + "Z",
        }, separators=(",", ":")).encode("utf-8")
        path_in_repo = f"manifests/{language}/{chunk_id}.json"
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                HF_API.create_commit(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    operations=[
                        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=payload),
                    ],
                    commit_message=f"Add manifest for {chunk_id}",
                )
                return
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    print(f"[manifest] ERROR committing {path_in_repo}: {e}")
                    return
                time.sleep(1 + attempts)
    except Exception as e:
        print(f"[manifest] ERROR building commit for {chunk_id}: {e}")


# --- 4. Main Orchestration Logic ---

def main():
    """
    Main function to orchestrate the sharding and uploading process.
    Handles two modes: --scan and process.
    """
    parser = argparse.ArgumentParser(description="Build and upload a Hugging Face dataset from R2.")
    parser.add_argument("--scan", action="store_true", help="Scan R2 and create a work plan. Run this once.")
    parser.add_argument("--limit-per-lang", type=int, default=0, help="Limit episodes per language during scan (0 = no limit).")
    parser.add_argument("--max-languages", type=int, default=0, help="Limit number of languages during scan (0 = no limit).")
    parser.add_argument("--max-rows-per-shard", type=int, default=MAX_ROWS_PER_SHARD, help="Max rows per Parquet shard.")
    parser.add_argument("--write-batch-rows", type=int, default=WRITE_BATCH_ROWS, help="Rows per Parquet write batch.")
    parser.add_argument("--download-workers", type=int, default=DOWNLOAD_WORKERS, help="Parallel metadata fetch workers.")
    parser.add_argument("--languages", type=str, default="", help="Comma-separated languages to process (e.g., 'en,de,ja'). Empty = all.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)

    if args.scan:
        print("--- Scan Mode ---")
        r2_client = R2_CLIENT
        limit = None if args.limit_per_lang <= 0 else args.limit_per_lang
        max_langs = None if args.max_languages <= 0 else args.max_languages
        prefixes_by_lang = list_all_episode_prefixes_by_language(r2_client, limit_per_lang=limit, max_languages=max_langs)
        print("[scan] Writing work plan (gz) to disk ...")
        os.makedirs(os.path.dirname(WORK_PLAN_FILE), exist_ok=True)
        with gzip.open(WORK_PLAN_FILE, 'wt', encoding='utf-8') as f:
            json.dump(prefixes_by_lang, f, indent=0, separators=(",", ":"))
        print(f"✅ Scan complete. Work plan saved to {WORK_PLAN_FILE}")
        return

    print("--- Process Mode ---")
    if not os.path.exists(WORK_PLAN_FILE):
        print(f"Error: Work plan '{WORK_PLAN_FILE}' not found.")
        print("Please run the script with the --scan flag first.")
        return

    print("[proc] Loading work plan ...")
    with gzip.open(WORK_PLAN_FILE, 'rt', encoding='utf-8') as f:
        prefixes_by_lang = json.load(f)
    print(f"[proc] Languages in plan: {len(prefixes_by_lang)}")

    if args.languages:
        requested = {
            _normalize_language_code(lang.strip())
            for lang in args.languages.split(",") if lang.strip()
        }
        if requested:
            prefixes_by_lang = {
                lang: prefixes for lang, prefixes in prefixes_by_lang.items()
                if _normalize_language_code(lang) in requested
            }
            print(f"[proc] Filtered to languages: {sorted(prefixes_by_lang.keys())}")
    
    completed_chunks = load_completed_chunks()
    print(f"[proc] Completed chunks in log: {len(completed_chunks)}")

    # We no longer create in-memory Dataset objects; we stream Parquet shards instead.

    for language, all_prefixes in prefixes_by_lang.items():
        print(f"\n===== Processing language: {language} (episodes={len(all_prefixes)}) =====")
        
        prefix_chunks = [
            all_prefixes[i:i + EPISODES_PER_CHUNK] 
            for i in range(0, len(all_prefixes), EPISODES_PER_CHUNK)
        ]
        print(f"[proc:{language}] chunks={len(prefix_chunks)} chunk_size<={EPISODES_PER_CHUNK}")

        for i, chunk_of_prefixes in enumerate(prefix_chunks):
            chunk_id = f"{language}-{i}"
            if chunk_id in completed_chunks:
                print(f"[proc:{language}] SKIP completed {chunk_id}")
                continue

            print(f"[proc:{language}] START chunk {chunk_id} episodes_in_chunk={len(chunk_of_prefixes)}")
            t_chunk = time.time()
            # Cross-process manifest check to avoid duplicate work
            manifest_path = f"manifests/{language}/{chunk_id}.json"
            if _repo_path_exists(manifest_path):
                print(f"[proc:{language}] SKIP (already manifested on HF) {chunk_id}")
                mark_chunk_as_completed(chunk_id)
                continue
            rows_written = 0
            shards_committed = 0
            os.makedirs(CHUNK_STATUS_DIR, exist_ok=True)
            status_path = os.path.join(CHUNK_STATUS_DIR, f"{chunk_id}.json")
            with open(status_path, 'w') as sf:
                json.dump({"chunk": chunk_id, "status": "in_progress", "ts": time.time()}, sf)

            writer = RollingParquetWriter(language, current_chunk=chunk_id, max_rows_per_shard=args.max_rows_per_shard, write_batch_rows=args.write_batch_rows)

            workers = min(args.download_workers, max(1, len(chunk_of_prefixes)))
            print(f"[proc:{language}] submitting {len(chunk_of_prefixes)} metadata jobs (workers={workers}) ...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_prefix = {executor.submit(fetch_episode_manifest_rows, prefix): prefix for prefix in chunk_of_prefixes}
                total_done = 0
                for future in concurrent.futures.as_completed(future_to_prefix):
                    rows = future.result()
                    total_done += 1
                    if total_done % 20 == 0 or total_done == len(chunk_of_prefixes):
                        print(f"[proc:{language}] progress {total_done}/{len(chunk_of_prefixes)} episodes processed")
                    if not rows:
                        continue
                    committed_locals = writer.write_rows(rows) or []
                    rows_written += len(rows)
                    for p in committed_locals:
                        if _commit_parquet_to_hub(p, language):
                            shards_committed += 1
                        try:
                            os.remove(p)
                        except Exception as e:
                            print(f"[cleanup] WARN failed to delete local shard {p}: {e}")

            # finalize last shard for this chunk
            last_local = writer.finalize()
            if last_local:
                if _commit_parquet_to_hub(last_local, language):
                    shards_committed += 1
                try:
                    os.remove(last_local)
                except Exception as e:
                    print(f"[cleanup] WARN failed to delete local shard {last_local}: {e}")
            writer.cleanup()

            if rows_written == 0 and shards_committed == 0:
                print(f"[proc:{language}] WARNING: no rows for {chunk_id}; will retry later.")
                with open(status_path, 'w') as sf:
                    json.dump({"chunk": chunk_id, "status": "empty", "ts": time.time()}, sf)
                continue

            # Commit per-chunk manifest and mark completed
            _commit_chunk_manifest(language, chunk_id, rows_written, shards_committed)
            mark_chunk_as_completed(chunk_id)
            with open(status_path, 'w') as sf:
                json.dump({
                    "chunk": chunk_id,
                    "status": "done",
                    "ts": time.time(),
                    "rows": rows_written,
                    "shards": shards_committed,
                }, sf)
            print(f"[proc:{language}] DONE {chunk_id} rows={rows_written} shards={shards_committed} (total {time.time()-t_chunk:.2f}s)")

    print("✅ All languages and chunks have been processed and uploaded!")


if __name__ == "__main__":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    required_vars = ["R2_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "HF_REPO_ID", "HF_TOKEN"]
    if any(v not in os.environ for v in required_vars):
        parser = argparse.ArgumentParser()
        parser.add_argument("--scan", action="store_true")
        args, _ = parser.parse_known_args()
        if not args.scan and "HF_TOKEN" not in os.environ:
            print("Error: Missing HF_TOKEN for process mode.")
            exit(1)
        elif any(v not in os.environ for v in required_vars if v != "HF_TOKEN"):
            print("Error: Missing one or more required R2 environment variables.")
            exit(1)

    main()

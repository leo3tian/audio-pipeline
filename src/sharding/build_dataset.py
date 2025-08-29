import os
import json
import boto3
import argparse
import concurrent.futures
import time
from datasets import Dataset, Features, Audio, Value
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from huggingface_hub import HfApi, CommitOperationAdd
import tempfile

# --- 1. Configuration ---
# pip install datasets "huggingface_hub[hf_transfer]" boto3
# Load from environment variables for security and flexibility
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

HF_REPO_ID = os.environ.get("HF_REPO_ID") # e.g., "your-org/your-50TB-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Batching and Concurrency Controls ---
# How many podcast episodes to process in one go before creating a shard and uploading.
# With 128GB RAM, you can use a large chunk size for more efficient commits.
EPISODES_PER_CHUNK = int(os.environ.get("EPISODES_PER_CHUNK", "1000"))
# Number of parallel threads to fetch data from R2.
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "128"))

# Max size per Parquet shard (bytes). Default ~2 GiB.
PARQUET_MAX_BYTES = int(os.environ.get("PARQUET_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))


# --- File Paths for State Management ---
# Store progress and plan in the process working directory (where the user runs the script)
BASE_STATE_DIR = os.getcwd()
PROGRESS_LOG = os.path.join(BASE_STATE_DIR, "progress.log")
WORK_PLAN_FILE = os.path.join(BASE_STATE_DIR, "work_plan.json")


# --- 2. R2 Client and Helper Functions ---

def get_r2_client():
    """Initializes and returns the Boto3 client for R2."""
    return boto3.client(
        service_name="s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

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

def list_all_episode_prefixes_by_language(r2_client, limit_per_lang=None, max_languages=None, languages_filter=None):
    """
    Efficiently lists unique episode prefixes from R2 grouped by normalized base language.
    Parallelizes episode listing per language to speed up scanning.
    """
    t0 = time.time()
    print("[scan] Listing all episode prefixes from R2 by language (parallel)...")
    if languages_filter:
        try:
            # Ensure set for O(1) lookup
            languages_filter = set(languages_filter)
        except Exception:
            pass
        print(f"[scan] Language filter active: {sorted(languages_filter)}")
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
            if languages_filter and normalized_language not in languages_filter:
                continue
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
        client = get_r2_client()
        collected = []
        try:
            ep_pages = client.get_paginator('list_objects_v2').paginate(
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


# --- 3. Parallel Data Fetching ---

def fetch_episode_data(prefix):
    """
    Fetches all data for a single episode (metadata and all audio segments).
    This function is designed to be run in a thread pool.
    """
    thread_local_client = get_r2_client()
    examples = []
    try:
        t0 = time.time()
        metadata_key = f"{prefix}all_segments.json"
        #print(f"[fetch] meta GET {metadata_key}")
        metadata_obj = thread_local_client.get_object(Bucket=R2_BUCKET, Key=metadata_key)
        segments = json.loads(metadata_obj['Body'].read())
        #print(f"[fetch] meta OK {metadata_key} segments={len(segments)} (+{time.time()-t0:.2f}s)")

        for i, segment in enumerate(segments):
            episode_id = prefix.strip('/').split('/')[-1]
            audio_filename = f"{episode_id}_{i:06d}.mp3"
            audio_key = f"{prefix}{audio_filename}"
            t1 = time.time()
            audio_obj = thread_local_client.get_object(Bucket=R2_BUCKET, Key=audio_key)
            audio_bytes = audio_obj['Body'].read()

            examples.append({
                "audio": {"path": audio_key, "bytes": audio_bytes},
                "text": segment.get("text", ""),
                "speaker_id": segment.get("speaker", "UNKNOWN"),
                "duration_seconds": segment.get("end", 0) - segment.get("start", 0),
                "dnsmos": segment.get("dnsmos", 0.0),
                "language": segment.get("language", "UNKNOWN"),
            })
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            print(f"[fetch] Warning: missing file under prefix {prefix}: {e}")
            return []
        print(f"[fetch] Warning: Boto3 ClientError on prefix {prefix}: {e}")
        return []
    except Exception as e:
        print(f"[fetch] Error processing prefix {prefix}: {e}")
        return []
    # print(f"[fetch] DONE {prefix} total_examples={len(examples)}")
    return examples


# --- 4. Main Orchestration Logic ---

def main():
    """
    Main function to orchestrate the sharding and uploading process.
    Handles two modes: --scan and process.
    """
    parser = argparse.ArgumentParser(description="Build and upload a Hugging Face dataset from R2.")
    parser.add_argument("--scan", action="store_true", help="Scan R2 and create a work plan. Run this once.")
    parser.add_argument(
        "--language", "--lang", dest="languages", action="append", default=None,
        help="Language code(s) to include (e.g., en, de, jp). Can be repeated or comma-separated."
    )
    args = parser.parse_args()

    # Build a normalized language filter set if provided
    languages_filter = None
    if args.languages:
        raw_tokens = []
        for value in args.languages:
            raw_tokens.extend([t.strip() for t in value.split(',') if t.strip()])
        normalized_tokens = [_normalize_language_code(t) for t in raw_tokens]
        # Filter out unknown / empty
        languages_filter = {t for t in normalized_tokens if t and t != "unknown"}

    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)

    if args.scan:
        print("--- Scan Mode ---")
        print("NOTE: Applying test limit of 10 episodes per language.")
        r2_client = get_r2_client()
        prefixes_by_lang = list_all_episode_prefixes_by_language(
            r2_client,
            languages_filter=languages_filter,
        )
        print("[scan] Writing work plan to disk ...")
        with open(WORK_PLAN_FILE, 'w') as f:
            json.dump(prefixes_by_lang, f, indent=2)
        print(f"✅ Scan complete. Work plan saved to {WORK_PLAN_FILE}")
        return

    print("--- Process Mode ---")
    if not os.path.exists(WORK_PLAN_FILE):
        print(f"Error: Work plan '{WORK_PLAN_FILE}' not found.")
        print("Please run the script with the --scan flag first.")
        return

    print("[proc] Loading work plan ...")
    with open(WORK_PLAN_FILE, 'r') as f:
        prefixes_by_lang = json.load(f)
    if languages_filter:
        print(f"[proc] Language filter active: {sorted(languages_filter)}")
        prefixes_by_lang = {
            k: v for k, v in prefixes_by_lang.items()
            if _normalize_language_code(k) in languages_filter
        }
    print(f"[proc] Languages in plan: {len(prefixes_by_lang)}")
    if not prefixes_by_lang:
        print("[proc] No languages to process after applying filter.")
        return
    
    completed_chunks = load_completed_chunks()
    print(f"[proc] Completed chunks in log: {len(completed_chunks)}")

    features = Features({
        'audio': Audio(sampling_rate=16000),
        'text': Value('string'),
        'speaker_id': Value('string'),
        'duration_seconds': Value('float32'),
        'dnsmos': Value('float32'),
        'language': Value('string'),
    })

    for language, all_prefixes in prefixes_by_lang.items():
        print(f"\n===== Processing language: {language} (episodes={len(all_prefixes)}) =====")
        
        prefix_chunks = [
            all_prefixes[i:i + EPISODES_PER_CHUNK] 
            for i in range(0, len(all_prefixes), EPISODES_PER_CHUNK)
        ]
        print(f"[proc:{language}] chunks={len(prefix_chunks)} chunk_size<=${EPISODES_PER_CHUNK}")

        # Set up a language-level progress tracker across all chunks
        language_done = 0
        lang_pbar = None
        if tqdm is not None:
            lang_pbar = tqdm(total=len(all_prefixes), desc=f"{language} episodes", mininterval=0.5)

        for i, chunk_of_prefixes in enumerate(prefix_chunks):
            chunk_id = f"{language}-{i}"
            if chunk_id in completed_chunks:
                print(f"[proc:{language}] SKIP completed {chunk_id}")
                continue

            print(f"[proc:{language}] START chunk {chunk_id} episodes_in_chunk={len(chunk_of_prefixes)}")
            t_chunk = time.time()
            
            all_examples_for_chunk = []
            print(f"[proc:{language}] submitting {len(chunk_of_prefixes)} fetch jobs (workers={DOWNLOAD_WORKERS}) ...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
                futures = [executor.submit(fetch_episode_data, prefix) for prefix in chunk_of_prefixes]
                if lang_pbar is not None:
                    for future in concurrent.futures.as_completed(futures):
                        examples = future.result()
                        if examples:
                            all_examples_for_chunk.extend(examples)
                        language_done += 1
                        lang_pbar.update(1)
                else:
                    for future in concurrent.futures.as_completed(futures):
                        examples = future.result()
                        language_done += 1
                        if language_done % 50 == 0:
                            print(f"[proc:{language}] progress {language_done}/{len(all_prefixes)} episodes fetched")
                        if examples:
                            all_examples_for_chunk.extend(examples)
            print(f"[proc:{language}] fetch complete ({len(all_examples_for_chunk)} segments) in {time.time()-t_chunk:.2f}s")
            
            if not all_examples_for_chunk:
                print(f"[proc:{language}] WARNING: no segments for {chunk_id}; marking complete to avoid re-run.")
                mark_chunk_as_completed(chunk_id)
                continue

            # Partition by approximate size so each Parquet file is <= PARQUET_MAX_BYTES
            def _partition_by_size(examples, max_bytes):
                groups = []
                current = []
                current_bytes = 0
                for ex in examples:
                    b = len(ex.get("audio", {}).get("bytes", b""))
                    if current and current_bytes + b > max_bytes:
                        groups.append(current)
                        current = []
                        current_bytes = 0
                    current.append(ex)
                    current_bytes += b
                if current:
                    groups.append(current)
                return groups

            parts = _partition_by_size(all_examples_for_chunk, PARQUET_MAX_BYTES)
            num_parts = len(parts)
            if num_parts == 1:
                print(f"[proc:{language}] building HF dataset object ...")
            else:
                print(f"[proc:{language}] building HF datasets for {num_parts} parts (size-limited)")

            # Append-only upload via create_commit: write Parquet per part
            base_repo_path = f"data/{language}/chunk-{i:05d}.parquet"
            print(f"[proc:{language}] uploading chunk {chunk_id} ({num_parts} part(s)) to {HF_REPO_ID} ...")
            try:
                t_up = time.time()
                operations = []
                tmp_paths = []
                api = HfApi()

                if num_parts == 1:
                    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmpf:
                        tmp_path = tmpf.name
                    tmp_paths.append(tmp_path)
                    Dataset.from_list(parts[0], features=features).to_parquet(tmp_path)
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=base_repo_path,
                            path_or_fileobj=tmp_path,
                        )
                    )
                else:
                    for p_idx, part_examples in enumerate(parts):
                        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmpf:
                            tmp_path = tmpf.name
                        tmp_paths.append(tmp_path)
                        Dataset.from_list(part_examples, features=features).to_parquet(tmp_path)
                        part_repo_path = f"data/{language}/chunk-{i:05d}-part-{p_idx:02d}.parquet"
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=part_repo_path,
                                path_or_fileobj=tmp_path,
                            )
                        )

                api.create_commit(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Append data chunk {i+1}/{len(prefix_chunks)} for lang '{language}' ({num_parts} part(s))",
                    token=HF_TOKEN,
                )

                for _p in tmp_paths:
                    try:
                        os.remove(_p)
                    except Exception:
                        pass
                mark_chunk_as_completed(chunk_id)
                print(f"[proc:{language}] DONE {chunk_id} upload in {time.time()-t_up:.2f}s (total {time.time()-t_chunk:.2f}s)")
            except Exception as e:
                print(f"[proc:{language}] ERROR uploading {chunk_id}: {e}")
                print(f"[proc:{language}] Will retry this chunk on next run.")

        if lang_pbar is not None:
            lang_pbar.close()

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
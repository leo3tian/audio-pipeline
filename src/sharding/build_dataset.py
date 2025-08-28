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
EPISODES_PER_CHUNK = int(os.environ.get("EPISODES_PER_CHUNK", "20000"))
# Number of parallel threads to fetch data from R2.
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "64"))


# --- File Paths for State Management ---
PROGRESS_LOG = "progress.log"
WORK_PLAN_FILE = "work_plan.json"


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

def list_all_episode_prefixes_by_language(r2_client, limit_per_lang=None, max_languages=None):
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
        print(f"[fetch] meta GET {metadata_key}")
        metadata_obj = thread_local_client.get_object(Bucket=R2_BUCKET, Key=metadata_key)
        segments = json.loads(metadata_obj['Body'].read())
        print(f"[fetch] meta OK {metadata_key} segments={len(segments)} (+{time.time()-t0:.2f}s)")

        for i, segment in enumerate(segments):
            episode_id = prefix.strip('/').split('/')[-1]
            audio_filename = f"{episode_id}_{i:06d}.mp3"
            audio_key = f"{prefix}{audio_filename}"
            t1 = time.time()
            print(f"[fetch] audio GET {audio_key}")
            audio_obj = thread_local_client.get_object(Bucket=R2_BUCKET, Key=audio_key)
            audio_bytes = audio_obj['Body'].read()
            print(f"[fetch] audio OK {audio_key} (+{time.time()-t1:.2f}s)")

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
    print(f"[fetch] DONE {prefix} total_examples={len(examples)}")
    return examples


# --- 4. Main Orchestration Logic ---

def main():
    """
    Main function to orchestrate the sharding and uploading process.
    Handles two modes: --scan and process.
    """
    parser = argparse.ArgumentParser(description="Build and upload a Hugging Face dataset from R2.")
    parser.add_argument("--scan", action="store_true", help="Scan R2 and create a work plan. Run this once.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)

    if args.scan:
        print("--- Scan Mode ---")
        print("NOTE: Applying test limit of 10 episodes per language.")
        r2_client = get_r2_client()
        prefixes_by_lang = list_all_episode_prefixes_by_language(r2_client, limit_per_lang=10)
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
    print(f"[proc] Languages in plan: {len(prefixes_by_lang)}")
    
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
                future_to_prefix = {executor.submit(fetch_episode_data, prefix): prefix for prefix in chunk_of_prefixes}
                total_done = 0
                for future in concurrent.futures.as_completed(future_to_prefix):
                    examples = future.result()
                    total_done += 1
                    if total_done % 5 == 0:
                        print(f"[proc:{language}] progress {total_done}/{len(chunk_of_prefixes)} episodes fetched")
                    if examples:
                        all_examples_for_chunk.extend(examples)
            print(f"[proc:{language}] fetch complete ({len(all_examples_for_chunk)} segments) in {time.time()-t_chunk:.2f}s")
            
            if not all_examples_for_chunk:
                print(f"[proc:{language}] WARNING: no segments for {chunk_id}; marking complete to avoid re-run.")
                mark_chunk_as_completed(chunk_id)
                continue

            print(f"[proc:{language}] building HF dataset object ...")
            dataset_chunk = Dataset.from_list(all_examples_for_chunk, features=features)

            print(f"[proc:{language}] uploading chunk {chunk_id} to {HF_REPO_ID} (config={language}) ...")
            try:
                t_up = time.time()
                dataset_chunk.push_to_hub(
                    repo_id=HF_REPO_ID,
                    config_name=language,
                    token=HF_TOKEN,
                    commit_message=f"Add data chunk {i+1}/{len(prefix_chunks)} for lang '{language}'"
                )
                mark_chunk_as_completed(chunk_id)
                print(f"[proc:{language}] DONE {chunk_id} upload in {time.time()-t_up:.2f}s (total {time.time()-t_chunk:.2f}s)")
            except Exception as e:
                print(f"[proc:{language}] ERROR uploading {chunk_id}: {e}")
                print(f"[proc:{language}] Will retry this chunk on next run.")

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
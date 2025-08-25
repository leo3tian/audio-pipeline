import os
import json
import boto3
import argparse
import concurrent.futures
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
EPISODES_PER_CHUNK = int(os.environ.get("EPISODES_PER_CHUNK", "1000"))
# Number of parallel threads to fetch data from R2.
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "4"))


# --- File Paths for State Management ---
PROGRESS_LOG = "src/sharding/progress.log"
WORK_PLAN_FILE = "src/sharding/work_plan.json"


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
    print("Listing all episode prefixes from R2 by language (parallel)...")
    prefixes_by_lang = {}
    paginator = r2_client.get_paginator('list_objects_v2')

    # 1) List language folders (e.g., 'processed/en/')
    lang_pages = paginator.paginate(Bucket=R2_BUCKET, Prefix="processed/", Delimiter='/')
    language_entries = []  # [(normalized_language, lang_prefix)]
    for page in lang_pages:
        for lang_prefix_data in page.get('CommonPrefixes', []):
            lang_prefix = lang_prefix_data.get('Prefix')
            raw_language = lang_prefix.strip('/').split('/')[-1]
            normalized_language = _normalize_language_code(raw_language)
            language_entries.append((normalized_language, lang_prefix))

    # Optionally cap number of languages for testing
    if max_languages is not None:
        language_entries = language_entries[:max_languages]

    # 2) Scan each language in parallel
    import concurrent.futures

    def scan_one_language(entry):
        normalized_language, lang_prefix = entry
        client = get_r2_client()
        collected = []
        try:
            ep_pages = client.get_paginator('list_objects_v2').paginate(
                Bucket=R2_BUCKET, Prefix=lang_prefix, Delimiter='/'
            )
            for ep_page in ep_pages:
                for ep_prefix in ep_page.get('CommonPrefixes', []):
                    collected.append(ep_prefix.get('Prefix'))
                    if limit_per_lang and len(collected) >= limit_per_lang:
                        return normalized_language, collected
        except Exception as e:
            print(f"  [scan] Warning: failed scanning language '{normalized_language}' at '{lang_prefix}': {e}")
        return normalized_language, collected

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for normalized_language, collected in executor.map(scan_one_language, language_entries):
            if not collected:
                continue
            existing = prefixes_by_lang.setdefault(normalized_language, [])
            existing.extend(collected)

    for lang, prefixes in prefixes_by_lang.items():
        print(f"  Found {len(prefixes)} episodes for language: '{lang}'")

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
    # Each thread needs its own client.
    thread_local_client = get_r2_client()
    examples = []
    try:
        # 1. Get the metadata for this episode
        metadata_key = f"{prefix}all_segments.json"
        metadata_obj = thread_local_client.get_object(Bucket=R2_BUCKET, Key=metadata_key)
        segments = json.loads(metadata_obj['Body'].read())

        # 2. For each segment, get the audio and create a structured record
        for i, segment in enumerate(segments):
            episode_id = prefix.strip('/').split('/')[-1]
            audio_filename = f"{episode_id}_{i:06d}.mp3"
            audio_key = f"{prefix}{audio_filename}"

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
            print(f"Warning: Skipping prefix {prefix} due to missing file: {e}")
            return [] # Return empty list on expected errors
        print(f"Warning: Boto3 ClientError on prefix {prefix}: {e}")
        return []
    except Exception as e:
        print(f"Error processing prefix {prefix}: {e}")
        return [] # Ensure we always return a list
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

    # Create the directory for state files if it doesn't exist
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)

    if args.scan:
        print("--- Scan Mode ---")
        # FOR TESTING: Limit the scan to a small number of episodes per language.
        # Remove `limit_per_lang=10` for the full run.
        print("NOTE: Applying test limit of 10 episodes per language.")
        r2_client = get_r2_client()
        prefixes_by_lang = list_all_episode_prefixes_by_language(r2_client, limit_per_lang=10)
        with open(WORK_PLAN_FILE, 'w') as f:
            json.dump(prefixes_by_lang, f, indent=2)
        print(f"✅ Scan complete. Work plan saved to {WORK_PLAN_FILE}")
        return

    print("--- Process Mode ---")
    if not os.path.exists(WORK_PLAN_FILE):
        print(f"Error: Work plan '{WORK_PLAN_FILE}' not found.")
        print("Please run the script with the --scan flag first.")
        return

    with open(WORK_PLAN_FILE, 'r') as f:
        prefixes_by_lang = json.load(f)
    
    completed_chunks = load_completed_chunks()

    # Define the structure of our final dataset
    features = Features({
        'audio': Audio(sampling_rate=16000),
        'text': Value('string'),
        'speaker_id': Value('string'),
        'duration_seconds': Value('float32'),
        'dnsmos': Value('float32'),
        'language': Value('string'),
    })

    # Process each language as a separate configuration
    for language, all_prefixes in prefixes_by_lang.items():
        print(f"\n===== Processing language: {language} =====")
        
        # Create chunks of work for the current language
        prefix_chunks = [
            all_prefixes[i:i + EPISODES_PER_CHUNK] 
            for i in range(0, len(all_prefixes), EPISODES_PER_CHUNK)
        ]
        print(f"Split '{language}' into {len(prefix_chunks)} chunks of up to {EPISODES_PER_CHUNK} episodes each.")

        # Process each chunk for the current language
        for i, chunk_of_prefixes in enumerate(prefix_chunks):
            chunk_id = f"{language}-{i}"
            if chunk_id in completed_chunks:
                print(f"--- Skipping already completed chunk {chunk_id} ({i+1}/{len(prefix_chunks)}) ---")
                continue

            print(f"--- Processing chunk {chunk_id} ({i+1}/{len(prefix_chunks)}) with {len(chunk_of_prefixes)} episodes ---")
            
            all_examples_for_chunk = []
            # Use ThreadPoolExecutor for high-performance parallel downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
                # Submit all fetch jobs to the pool
                future_to_prefix = {executor.submit(fetch_episode_data, prefix): prefix for prefix in chunk_of_prefixes}
                
                for future in concurrent.futures.as_completed(future_to_prefix):
                    # The result of fetch_episode_data is a list of examples
                    examples = future.result()
                    if examples:
                        all_examples_for_chunk.extend(examples)
            
            if not all_examples_for_chunk:
                print(f"Warning: No valid segments found for chunk {chunk_id}. Skipping.")
                # Mark as complete even if empty to avoid reprocessing a known-bad chunk
                mark_chunk_as_completed(chunk_id)
                continue

            print(f"Generated dataset for chunk {chunk_id} with {len(all_examples_for_chunk)} segments.")
            
            # Create the dataset object from the in-memory list of examples
            dataset_chunk = Dataset.from_list(all_examples_for_chunk, features=features)

            # Push this chunk to the Hugging Face Hub under the language-specific config
            print(f"Uploading chunk {chunk_id} to {HF_REPO_ID} (config: {language})...")
            try:
                dataset_chunk.push_to_hub(
                    repo_id=HF_REPO_ID,
                    config_name=language,
                    token=HF_TOKEN,
                    commit_message=f"Add data chunk {i+1}/{len(prefix_chunks)} for lang '{language}'"
                )
                # If and only if the upload succeeds, mark the chunk as complete.
                mark_chunk_as_completed(chunk_id)
                print(f"--- Successfully completed and uploaded chunk {chunk_id} ---")
            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"CRITICAL: Failed to upload chunk {chunk_id}. Error: {e}")
                print(f"This chunk will be retried on the next run.")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    print("✅ All languages and chunks have been processed and uploaded!")


if __name__ == "__main__":
    # Ensure hf-transfer is enabled for maximum speed
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Basic check for required environment variables
    required_vars = ["R2_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "HF_REPO_ID", "HF_TOKEN"]
    if any(v not in os.environ for v in required_vars):
        # Allow scan mode without HF token
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

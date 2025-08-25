import os
import json
import boto3
import tempfile
import itertools
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
# Tune this based on your tinybox's RAM. 1000 is a safe starting point.
EPISODES_PER_CHUNK = int(os.environ.get("EPISODES_PER_CHUNK", "1000"))

# File to track which chunks we've already processed
PROGRESS_LOG = "src/sharding/progress.log"

# Global client placeholder (avoid storing actual client in module globals)
r2_client = None


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

def list_all_episode_prefixes_by_language(r2_client, limit_per_lang=None):
    """
    Efficiently lists all unique episode prefixes from R2 and groups them by language.
    Returns a dictionary like: {'en': ['path/to/en/ep1', ...], 'es': ['path/to/es/ep1', ...]}
    """
    print("Listing all episode prefixes from R2 by language...")
    prefixes_by_lang = {}
    paginator = r2_client.get_paginator('list_objects_v2')
    
    # First, get all language folders (e.g., 'processed/en/')
    lang_pages = paginator.paginate(Bucket=R2_BUCKET, Prefix="processed/", Delimiter='/')
    
    for page in lang_pages:
        for lang_prefix_data in page.get('CommonPrefixes', []):
            lang_prefix = lang_prefix_data.get('Prefix')
            # Extract language code, e.g., 'en' from 'processed/en/'
            raw_language = lang_prefix.strip('/').split('/')[-1]
            normalized_language = _normalize_language_code(raw_language)

            # Now, for each language, get all episode folders
            ep_pages = paginator.paginate(Bucket=R2_BUCKET, Prefix=lang_prefix, Delimiter='/')
            for ep_page in ep_pages:
                for ep_prefix in ep_page.get('CommonPrefixes', []):
                    # Use setdefault to initialize the list if the key is new
                    prefixes_by_lang.setdefault(normalized_language, []).append(ep_prefix.get('Prefix'))

    # After aggregating, apply the limit if one was provided
    if limit_per_lang:
        print(f"Applying limit of {limit_per_lang} episodes per language for testing.")
        for lang in prefixes_by_lang:
            prefixes_by_lang[lang] = prefixes_by_lang[lang][:limit_per_lang]

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


# --- 3. The Data Generator ---
# This is the memory-efficient core of the script.

def generate_examples(episode_prefixes):
    """
    A Python generator that yields one processed example at a time.
    It streams data directly from R2, processes it, and yields.
    This avoids loading the entire dataset into RAM.
    """
    client = get_r2_client()
    for prefix in episode_prefixes:
        try:
            # 1. Get the metadata for this episode
            metadata_key = f"{prefix}all_segments.json"
            metadata_obj = client.get_object(Bucket=R2_BUCKET, Key=metadata_key)
            segments = json.loads(metadata_obj['Body'].read())

            # 2. For each segment, get the audio and yield a structured record
            for i, segment in enumerate(segments):
                # Correctly handle episode ID extraction from prefix
                episode_id = prefix.strip('/').split('/')[-1]
                audio_filename = f"{episode_id}_{i:06d}.mp3"
                audio_key = f"{prefix}{audio_filename}"

                audio_obj = client.get_object(Bucket=R2_BUCKET, Key=audio_key)
                audio_bytes = audio_obj['Body'].read()

                yield {
                    "audio": {"path": audio_key, "bytes": audio_bytes},
                    "text": segment.get("text", ""),
                    "speaker_id": segment.get("speaker", "UNKNOWN"),
                    "duration_seconds": segment.get("end", 0) - segment.get("start", 0),
                    "dnsmos": segment.get("dnsmos", 0.0),
                    "language": segment.get("language", "UNKNOWN"),
                }
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                print(f"Warning: Skipping prefix {prefix} due to missing file: {e}")
                continue
            # For other AWS errors, print and continue
            print(f"Warning: Boto3 ClientError on prefix {prefix}: {e}")
            continue
        except Exception as e:
            print(f"Error processing prefix {prefix}: {e}")
            continue


# --- 4. Main Orchestration Logic ---

def main():
    """
    Main function to orchestrate the sharding and uploading process.
    """
    # Create the directory for the progress log if it doesn't exist
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
    
    # Create a local client for listing prefixes; avoid storing in module globals
    r2_client = get_r2_client()
    completed_chunks = load_completed_chunks()

    # Get the full list of work to be done, grouped by language
    # FOR TESTING: Only process the first 1 episode per language. Remove `limit_per_lang=1` for a full run.
    prefixes_by_lang = list_all_episode_prefixes_by_language(r2_client, limit_per_lang=1)

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
        for i, chunk in enumerate(prefix_chunks):
            chunk_id = f"{language}-{i}"
            if chunk_id in completed_chunks:
                print(f"--- Skipping already completed chunk {chunk_id} ({i+1}/{len(prefix_chunks)}) ---")
                continue

            print(f"--- Processing chunk {chunk_id} ({i+1}/{len(prefix_chunks)}) ---")
            
            # Use the generator to create a Dataset object for this chunk
            dataset_chunk = Dataset.from_generator(
                generate_examples,
                features=features,
                gen_kwargs={"episode_prefixes": chunk},
            )
            
            print(f"Generated dataset for chunk {chunk_id} with {len(dataset_chunk)} segments.")

            # Push this chunk to the Hugging Face Hub under the language-specific config
            print(f"Uploading chunk {chunk_id} to {HF_REPO_ID} (config: {language})...")
            dataset_chunk.push_to_hub(
                repo_id=HF_REPO_ID,
                config_name=language, # This is the key change!
                token=HF_TOKEN,
                commit_message=f"Add data chunk {i+1}/{len(prefix_chunks)} for lang '{language}'"
            )
            
            # Mark as complete and move to the next
            mark_chunk_as_completed(chunk_id)
            print(f"--- Successfully completed chunk {chunk_id} ---")

    print("✅ All languages and chunks have been processed and uploaded!")


if __name__ == "__main__":
    # Ensure hf-transfer is enabled for maximum speed
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # Basic check for required environment variables
    required_vars = ["R2_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "HF_REPO_ID", "HF_TOKEN"]
    if any(v not in os.environ for v in required_vars):
        print("Error: Missing one or more required environment variables.")
        print("Please set: R2_BUCKET, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, HF_REPO_ID, HF_TOKEN")
        exit(1)

    main()

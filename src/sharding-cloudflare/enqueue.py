import os
import json
import boto3
import argparse
import concurrent.futures
import time
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv()

# Cloudflare SDK is now a hard requirement
from cloudflare import Cloudflare, APIError

# --- 1. Configuration ---
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

CF_ACCOUNT_ID = os.environ.get("CF_ID")
CF_API_TOKEN = os.environ.get("CF_TOKEN")
CF_QUEUE_NAME = os.environ.get("CF_QUEUE_NAME")
CF_QUEUE_ID_ENV = os.environ.get("CF_QUEUE_ID")

# --- Batching and Concurrency Controls ---
EPISODES_PER_PLAN = int(os.environ.get("EPISODES_PER_PLAN", "500"))
MAX_EPISODES_PER_LANGUAGE = 10
ENQUEUE_WORKERS = int(os.environ.get("ENQUEUE_WORKERS", "64"))
MAX_LANGUAGES = 2

# --- State Management ---
PROGRESS_LOG = "src/sharding-cloudflare/enqueue_progress.log"
R2_PLAN_PREFIX = "tasks/cf-upload-plans/"
CF_QUEUE_ID_CACHE = "src/sharding-cloudflare/.queue_id"


# --- 2. Helper Functions ---

def get_r2_client():
    """Initializes a thread-safe Boto3 client for R2."""
    return boto3.client(
        service_name="s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, max_pool_connections=ENQUEUE_WORKERS),
    )

def _normalize_language_code(lang_str):
    """Normalizes a messy language folder name into a standard base code."""
    if not lang_str: return "unknown"
    code = lang_str.lower().strip()
    if code in ['deu', 'ger']: return 'de'
    if code == 'eng': return 'en'
    if code == 'srp': return 'sr'
    if code == 'in': return 'id'
    if code in ['unite', 'un', 'unknown', '']: return 'unknown'
    return code.split('-')[0].split('_')[0]

def list_all_episode_prefixes_by_language(r2_client):
    """Parallel-scans R2 to find all episode prefixes, grouped by language."""
    print("Listing episode prefixes from R2 (parallel)...")
    prefixes_by_lang = {}
    paginator = r2_client.get_paginator('list_objects_v2')

    lang_pages = paginator.paginate(Bucket=R2_BUCKET, Prefix="processed/", Delimiter='/')
    language_entries = []
    for page in lang_pages:
        for lang_prefix_data in page.get('CommonPrefixes', []):
            lang_prefix = lang_prefix_data.get('Prefix')
            raw_language = lang_prefix.strip('/').split('/')[-1]
            normalized_language = _normalize_language_code(raw_language)
            language_entries.append((normalized_language, lang_prefix))

    if MAX_LANGUAGES is not None:
        language_entries = language_entries[:MAX_LANGUAGES]
        print(f"Applying MAX_LANGUAGES cap: {len(language_entries)} language(s)")

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
                    if MAX_EPISODES_PER_LANGUAGE and len(collected) >= MAX_EPISODES_PER_LANGUAGE:
                        return normalized_language, collected
        except Exception as e:
            print(f"Warning: failed scanning language '{normalized_language}': {e}")
        return normalized_language, collected

    with concurrent.futures.ThreadPoolExecutor(max_workers=ENQUEUE_WORKERS) as executor:
        for normalized_language, collected in executor.map(scan_one_language, language_entries):
            if collected:
                prefixes_by_lang.setdefault(normalized_language, []).extend(collected)

    print("Scan complete.")
    for lang, prefixes in prefixes_by_lang.items():
        print(f"  Found {len(prefixes)} episodes for language: '{lang}'")
    return prefixes_by_lang

def load_completed_plans():
    """Loads the set of plan keys that have already been enqueued."""
    if not os.path.exists(PROGRESS_LOG):
        return set()
    with open(PROGRESS_LOG, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def mark_plan_as_completed(plan_key):
    """Appends a completed plan key to the progress log."""
    with open(PROGRESS_LOG, 'a') as f:
        f.write(f"{plan_key}\n")

def publish_to_cf_queue(cf_client, queue_id: str, message_body: dict):
    """Publishes a single message to the configured Cloudflare Queue using the SDK."""
    try:
        # Per SDK error, body is expected to be a JSON-serializable object, not a string.
        cf_client.queues.messages.bulk_push(
            account_id=CF_ACCOUNT_ID,
            queue_id=queue_id,
            messages=[{"body": message_body}],
        )
        return True
    except APIError as e:
        print(f"Error: Cloudflare API error during publish: {e.message}")
    except Exception as e:
        print(f"Error: Unexpected error during publish: {e}")
    return False

def resolve_cf_queue_id(cf_client) -> str:
    """Resolve the Cloudflare Queue ID from env, cache, or API using the SDK."""
    if CF_QUEUE_ID_ENV:
        return CF_QUEUE_ID_ENV

    if os.path.exists(CF_QUEUE_ID_CACHE):
        with open(CF_QUEUE_ID_CACHE, 'r') as f:
            cached_id = f.read().strip()
            if cached_id:
                return cached_id

    print(f"Queue ID not in env or cache, querying API for queue named '{CF_QUEUE_NAME}'...")
    try:
        resp = cf_client.queues.list(account_id=CF_ACCOUNT_ID)
        for q in resp:
            if q.name == CF_QUEUE_NAME:
                qid = q.id
                with open(CF_QUEUE_ID_CACHE, 'w') as f:
                    f.write(qid)
                return qid
    except APIError as e:
        raise RuntimeError(f"Cloudflare API error while listing queues: {e.message}")
    
    raise RuntimeError(f"Could not find a queue named '{CF_QUEUE_NAME}'. Please check the name or set CF_QUEUE_ID.")

# --- 3. Main Orchestration Logic ---

def create_and_enqueue_plan(plan_details, queue_id: str):
    """
    Core work unit: creates a plan file, uploads it, and enqueues a message.
    """
    plan_key, language, prefixes = plan_details
    # Each thread needs its own clients
    r2_client = get_r2_client()
    cf_client = Cloudflare(api_token=CF_API_TOKEN)
    
    try:
        plan_content = "\n".join(prefixes)
        
        r2_client.put_object(
            Bucket=R2_BUCKET,
            Key=plan_key,
            Body=plan_content.encode('utf-8')
        )

        message = {
            "plan_key": plan_key,
            "language": language,
            "shard_id_prefix": os.path.splitext(os.path.basename(plan_key))[0]
        }
        
        if not publish_to_cf_queue(cf_client, queue_id, message):
            print(f"Failed to enqueue message for {plan_key}. Will retry on next run.")
            return None

        return plan_key
    except Exception as e:
        print(f"Error processing plan {plan_key}: {e}")
        return None

def main():
    """
    Main function to scan R2, create plan files, and enqueue tasks.
    """
    os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
    
    # Initialize a single client for main-thread tasks
    r2_client = get_r2_client()
    cf_client = Cloudflare(api_token=CF_API_TOKEN)

    completed_plans = load_completed_plans()
    print(f"Found {len(completed_plans)} previously completed plans in {PROGRESS_LOG}.")

    prefixes_by_lang = list_all_episode_prefixes_by_language(r2_client)
    
    queue_id = resolve_cf_queue_id(cf_client)
    print(f"Using Cloudflare Queue ID: {queue_id}")

    all_plan_details = []
    for language, prefixes in prefixes_by_lang.items():
        chunks = [
            prefixes[i:i + EPISODES_PER_PLAN]
            for i in range(0, len(prefixes), EPISODES_PER_PLAN)
        ]
        for i, chunk_prefixes in enumerate(chunks):
            plan_key = f"{R2_PLAN_PREFIX}{language}-{i}.jsonl"
            if plan_key not in completed_plans:
                all_plan_details.append((plan_key, language, chunk_prefixes))
    
    if not all_plan_details:
        print("✅ No new plans to create. Everything is already enqueued.")
        return

    print(f"Found {len(all_plan_details)} new plans to create and enqueue...")
    
    successful_plans = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=ENQUEUE_WORKERS) as executor:
        future_to_plan = {
            executor.submit(create_and_enqueue_plan, details, queue_id): details 
            for details in all_plan_details
        }
        for future in concurrent.futures.as_completed(future_to_plan):
            result_key = future.result()
            if result_key:
                mark_plan_as_completed(result_key)
                successful_plans += 1
                print(f"Successfully enqueued plan: {result_key} ({successful_plans}/{len(all_plan_details)})")

    print(f"✅ Enqueuer run finished. Successfully processed {successful_plans}/{len(all_plan_details)} new plans.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan R2 and enqueue sharding tasks for Cloudflare Workers.")
    args = parser.parse_args()
    
    required_vars = ["R2_BUCKET", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", 
                     "CF_ID", "CF_TOKEN", "CF_QUEUE_NAME"]
    if any(v not in os.environ for v in required_vars):
        print("Error: Missing one or more required environment variables.")
        exit(1)

    main()

# Podcast Downloader

Two components to:
1) Read RSS feeds from the PodcastIndex database, crawl RSS to find episode URLs, and enqueue them for download
2) Download audio files using those episode URLs

Notes
- An RSS feed is basically a file that summarizes a podcast. We're interested in RSS feeds because they contain the audio enclosures of every episode in the podcast (which we can then download)
- The PodcastIndex database is a database of 4 million podcasts and their RSS feeds. The goal here is to download episodes from the podcasts of the PodcastIndex database.

## File Breakdown & Infrastructure

At a high level, the pipeline moves from PodcastIndex database → RSS worker → episode URL downloader. There is a diagram at the bottom of this file if it's helpful.

- **PodcastIndex database (Postgres)**: The PodcastIndex database holds RSS URLs and acts as the source of truth for job status. Podcasts are marked as `pending → in_progress → complete/failed` as they are processed.
- `worker.py` reads pending feeds directly from Postgres using `FOR UPDATE SKIP LOCKED`, marks them `in_progress`, fetches/parses RSS, filters non-dialogue content, checks DynamoDB for already-processed episodes, and enqueues new episode URLs to a **Cloudflare Queue**.
- **DynamoDB database:** Tracks all downloaded episodes. We check this before enqueueing to avoid duplicates.
- **Cloudflare downloader:** `cf-downloader/index.js` runs as a Cloudflare Worker, consuming episode URLs from the queue, downloading them, then saving them to R2. It also records episode URLs in DynamoDB to prevent duplicates.

## Setup

This section includes information about how to reset the pipeline.

### How to reset the pipeline

If you need to re‑run everything (download from scratch), reset these components:

1) Reset Postgres (The PodcastIndex database): this marks all feeds unprocessed. See next section for instructions.
2) Purge your Cloudflare queue (via the dashboard).
3) Recreate DynamoDB table: delete and recreate `PodcastIndexJobs` with primary key `episode_url`.
4) Clear storage: delete R2 objects (or bucket).

Once reset, run `worker.py` to repopulate and process from scratch.

### How to reset Postgres (The PodcastIndex database)

The original PodcastIndex database is in SQLite. We've converted it to Postgres for ease of use. The following steps will recreate the table and load the data.

0) Download the SQLite database from the PodcastIndex website (https://podcastindex.org/, scroll near the bottom under "Developer? Join the fun!" to find a download link).

1) Export the minimal columns from SQLite (run where `podcastindex.db` lives):

```bash
sqlite3 -csv podcastindex.db "SELECT id, url, language FROM podcasts;" > podcasts.csv
```

2) Connect to Postgres and create the table:

```bash
psql -h $PG_HOST -U $PG_USER -d $PG_DATABASE
```

```sql
DROP TABLE IF EXISTS podcasts;
CREATE TABLE podcasts (
    id BIGINT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    language TEXT,
    processing_status TEXT DEFAULT 'pending' NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

3) Bulk import the CSV (client‑side copy, runs from your machine/EC2):

```sql
\COPY podcasts (id, url, language) FROM 'podcasts.csv' WITH (FORMAT csv);
```

### How to Run

These components are designed to be run on EC2 instances with access to RDS, DynamoDB, and Cloudflare.

### 1. Running `worker.py` on AWS EC2

`worker.py` must be on an EC2 instance that can access your RDS instance, DynamoDB, and Cloudflare.

#### Environment variables

Set these for `worker.py`:

- **Database**: `PG_HOST`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`
- **Queues**:
  - Cloudflare (`worker.py` → CF): `CF_ACCOUNT_ID`, `CF_QUEUE_ID`, `CF_API_TOKEN`
- **DynamoDB**: `DYNAMODB_TABLE_NAME`
- **General**: `AWS_REGION`
- **Tuning** (optional): `FEEDER_BATCH_SIZE`, `FEEDER_SLEEP_SECONDS`, `STALE_JOB_TIMEOUT_MINUTES`, `DB_UPDATE_BATCH_SIZE`

`downloader.py` requires `SQS_QUEUE_URL` and `S3_BUCKET_NAME`.

### 2. Running the Cloudflare Downloader (`cf-downloader/index.js`)

The CF downloader is a Worker that consumes jobs from a Cloudflare Queue.

- **Configuration**:
  - Edit `wrangler.toml` to bind your R2 bucket and Cloudflare Queue.
  - Set AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) and `DYNAMODB_TABLE_NAME` as secrets: `npx wrangler secret put <NAME>`
- **Deployment**:
  - Run `npm install` inside `/cf-downloader`.
  - Deploy with `npx wrangler deploy`.
  - View live logs with `npx wrangler tail`.

### Diagram

```
[ Postgres DB: Feed URLs & Status ]
      |
      | 1. worker.py reserves 'pending' feeds (SKIP LOCKED)
      v
[ worker.py Fleet ]-------------------------------------+
      |                                                 |
      | 2. Parses feed, finds episode URLs,             | 3. Updates feed status in DB
      |    and enqueues them to Cloudflare Queue        |    ('complete' or 'failed')
      v                                                 v
[ Cloudflare Queue ]                         [ Postgres DB ]
(Episode URLs)
      |
      | 4. CF Worker consumes
      v
[ cf-downloader ]
      |
      | 5. Streams audio to R2
      v
[ Cloudflare R2 ]
(Final Storage)
```

### Optional: Amazon S3 download script

If you prefer to download to Amazon S3 instead of R2, there is a simple legacy script you can use:

- Script: `src/downloading/podcastindex/downloader.py`
- Expected input: episode URLs via an AWS SQS queue
- Required env vars: `SQS_QUEUE_URL`, `S3_BUCKET_NAME`, and standard AWS credentials

Note: This path is not part of the main pipeline. You would need to route episode URLs to SQS yourself and manage S3 storage/reset independently.
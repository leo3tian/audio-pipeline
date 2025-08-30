# Podcast Downloader

Three scripts to: 
1) Read RSS feeds from the PodcastIndex database 
2) Crawl RSS feeds to find episode URLs 
3) Download audio files using episode URLs

Notes
- An RSS feed is basically a file that summarizes a podcast. We're interested in RSS feeds because they contain the audio enclosures of every episode in the podcast (which we can then download)
- The PodcastIndex database is a database of 4 million podcasts and their RSS feeds. The goal here is to download episodes from the podcasts of the PodcastIndex database.

## File Breakdown & Infrastructure

At a high level, the pipeline moves from PodcastIndex database  → RSS feed queue → RSS feed crawler → episode URL queue → episode URL downloader. There is a diagram at the bottom of this file if it's helpful.

- **PodcastIndex database (Postgres)**: The PodcastIndex database which holds RSS URLs and also acts as the source of truth for job status. Podcasts are marked as `pending → in_progress → complete/failed` as they are processed. To interact with it, we've converted it to a PostgreSQL and hosted it on AWS RDS - see Setup section for how that works.
- `feeder.py` reads podcast RSS feeds from Postgres and enqueues the feeds to an AWS SQS queue, marking them as in-progress.
- `worker.py` consumes RSS feeds from the AWS SQS queue, crawls the RSS feeds to find all download URLs, then enqueues to **either a Cloudflare Queue or AWS SQS**
    - It also filters non-dialogue content, checks DynamoDB for already-processed episodes, and enqueues new episode URLs to the download queue. By default it pushes to a Cloudflare Queue, but alternatively you can push to AWS SQS and consume with `downloader.py`.
- **DynamoDB database:** Database that keeps track of all downloaded episodes. Before downloading an episode, we check this database to make sure no episodes are repeated
>**Depending on whether you choose Cloudflare R2 or AWS S3 as your final data destination, you will use one of the following scripts.**

- **Cloudflare:** `cf-downloader/index.js` runs as a Cloudflare Worker, consuming episode URLs from the queue, downloading them, then saving them to R2.
    - It also records episode URLs in DynamoDB to prevent duplicates.

- **AWS:** `downloader.py` is a downloader that consumes from `SQS_QUEUE_URL`, streams audio to S3 (`S3_BUCKET_NAME`), and writes a dedupe record to DynamoDB.

>*Q: Why not just have one script for both R2 and S3?*
>
>*A: Egress fees - if we only had a downloader on AWS, we would pay hundreds of dollars per TB downloaded when uploading to Cloudflare*
>For that reason, **Cloudflare is heavily recommended** - zero egress fees when pulling data to process on GPU clusters or upload to huggingface 

## Setup

### How to start the Cloudflare worker

Deploy worker: (from /podcastindex) run `npx wrangler deploy`

Tail worker logs: `npx wrangler tail`

### How to reset the pipeline

If you need to re‑run everything (download from scratch), reset these five components:

1) Reset Postgres (The PodcastIndex database): this marks all feeds unprocessed. See next section for instructions.
2) Purge your Cloudflare queue (via the dashboard).
3) Purge your AWS SQS queues (`FeedsToProcessQueue`, `PodcastIndexQueue`).
4) Recreate DynamoDB table: delete and recreate `PodcastIndexJobs` with primary key `episode_url`.
5) Clear storage: delete R2 objects (or bucket) or S3 prefix used for downloads.

Once reset, run `feeder.py` then `worker.py` to repopulate and process from scratch.

### How to move the PodcastIndex DB to PostgreSQL

Load the PodcastIndex dataset (SQLite) straight into Postgres using CSV and `psql` (fast, no Python required).

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

These scripts are designed to be run on separate EC2 instances.

### 1. Running `feeder.py` and `worker.py` on AWS EC2

`feeder.py` and `worker.py` must be on an EC2 instance that can access your RDS instance, SQS, and DynamoDB.

#### Environment variables

Set these for `feeder.py` and `worker.py`:

- **Database**: `PG_HOST`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`
- **Queues**:
  - `FEEDS_SQS_QUEUE_URL` (for `feeder.py` → `worker.py`)
  - Cloudflare (`worker.py` → CF): `CF_ACCOUNT_ID`, `CF_QUEUE_ID`, `CF_API_TOKEN`
  - SQS (`worker.py` → `downloader.py`): `SQS_QUEUE_URL`
- **DynamoDB**: `DYNAMODB_TABLE_NAME`
- **General**: `AWS_REGION`
- **Tuning** (optional): `FEEDER_BATCH_SIZE`, `FEEDER_SLEEP_SECONDS`, `STALE_JOB_TIMEOUT_MINUTES`, `DB_UPDATE_BATCH_SIZE`

`downloader.py` requires `SQS_QUEUE_URL` and `S3_BUCKET_NAME`.

### Running the Cloudflare Downloader (`cf-downloader/index.js`)

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
      | 1. Feeder reads 'pending' feeds
      v
[ feeder.py ]
      |
      | 2. Enqueues jobs to be processed
      v
[ SQS Feed Queue ]
      |
      | 3. Workers consume feed jobs
      v
[ worker.py Fleet ]-------------------------------------+
      |                                                 |
      | 4. Parses feed, finds episode URLs,             | 5. Updates feed status in DB
      |    and enqueues them into ONE of two paths:     |    ('complete' or 'failed')
      v                                                 v
+---------------------------------+           [ Postgres DB ]
|         PIPELINE SPLITS         |
+---------------------------------+
                 |
  +--------------+-----------------+
  |                                |
  v                                v
[ Cloudflare Queue ]        [ SQS Download Queue ]
(Episode URLs)              (Episode URLs)
  |                                |
  | 6a. CF Worker consumes         | 6b. EC2 Fleet consumes
  v                                v
[ cf-downloader ]           [ downloader.py ]
  |                                |
  | 7a. Streams audio to R2        | 7b. Streams audio to S3
  v                                v
[ Cloudflare R2 ]           [ AWS S3 ]
(Final Storage)             (Final Storage)
```
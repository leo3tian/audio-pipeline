# Utility Scripts

A collection of helper scripts for inspecting data feeds and managing files in R2/S3.

---

### `inspect_feed.py`

**Purpose:**
This tool fetches and parses a remote RSS feed. It's primarily used for debugging and validating podcast feeds from PodcastIndex. It displays the feed's metadata and, crucially, calculates a "dialogue score" based on predefined criteria (categories, keywords) to predict whether the feed contains spoken content or music. This helps determine if a feed passes the project's content filters.

**How to Use:**
Provide the RSS feed URL as a command-line argument.

```bash
python util/inspect_feed.py "http://example.com/podcast.rss"
```

The script will output the feed's score, whether it would be included or filtered, and a structured printout of the feed's and first episode's metadata.

---

### `r2_sampler.py`

**Purpose:**
This script efficiently samples random audio files from a specified prefix in a Cloudflare R2 bucket. Instead of listing all files (which can be slow and costly for millions of objects), it jumps to random points in the key space to find samples. It then generates temporary, pre-signed URLs for these files, which can be opened directly in a browser for listening. This is useful for quick, random quality checks or for getting a sense of the average duration and content of the audio files in the dataset.

**How to Use:**
First, set your Cloudflare R2 credentials in a `.env` file or as environment variables.

```env
# .env file
R2_BUCKET_NAME="your-r2-bucket"
CF_ID="your-cloudflare-account-id"
R2_ID="your-r2-access-key-id"
R2_KEY="your-r2-secret-access-key"
```

Then, run the script. You can configure the `R2_PREFIX` and `SAMPLE_SIZE` inside the script itself.

```bash
python util/r2_sampler.py
```

---

### `s3_sampler.py`

**Purpose:**
Similar to the R2 sampler, this script efficiently samples random audio files from an AWS S3 bucket. The key difference is that instead of generating URLs, it downloads the sampled files to a local `samples/` directory for inspection. This is useful when you need to analyze the files locally with audio tools. The script will clean up the downloaded files upon exit.

**How to Use:**
Ensure your AWS credentials are configured (e.g., via `~/.aws/credentials`). Then, run the script. The S3 bucket, prefix, and sample size can be configured inside the script.

```bash
python util/s3_sampler.py
```

---

### `r2_tools.py`

**Purpose:**
A command-line tool for managing and quantifying data in a Cloudflare R2 bucket. It provides subcommands to count files or folders and to move objects between prefixes. This is essential for monitoring the progress of large-scale download or processing jobs (e.g., counting completed tasks) and for administrative tasks like re-queueing stuck items by moving them from an `in_progress` prefix back to a `todo` prefix.

**How to Use:**
Set your Cloudflare R2 credentials in a `.env` file first. The tool is invoked with a command (`count`, `list-folders`, `move`) and relevant arguments.

**Examples:**

*   **Count completed processing tasks:**
    ```bash
    python util/r2_tools.py --bucket podcastindex-dataset count --prefix tasks/processing_completed/ --kind files
    ```

*   **Count immediate subfolders in a prefix:**
    ```bash
    python util/r2_tools.py --bucket my-bucket count --prefix raw_audio/ --kind folders
    ```

*   **Recursively count all files in a prefix:**
    ```bash
    python util/r2_tools.py --bucket my-bucket count --prefix raw_audio/en/ --kind files --recursive
    ```

*   **Move 100 stuck tasks from `in_progress` back to `todo`:**
    ```bash
    python util/r2_tools.py --bucket my-bucket move --from tasks/processing_in_progress/ --to tasks/processing_todo/ --limit 100
    ```

*   **Do a dry run to see which files would be moved:**
    ```bash
    python util/r2_tools.py --bucket my-bucket move --from old/prefix/ --to new/prefix/ --dry-run
    ```

## Overview
This repo builds an audio dataset in three steps:

1) Downloading 
2) Processing 
3) Uploading 

Each major folder has its own README.md file with more detailed information.

### Notes

As of 09/02/2025, the flow I'm using is downloading from PodcastIndex to R2, then processing and uploading to Hugging Face. The following folders are used the most:
- `src/downloading/podcastindex/`
- `src/processing/`
- `src/uploading/r2/`

Util scripts are in `util/` and can be used to gauge the progress of the pipeline via reading from Cloudflare R2.

## Other files
- Python deps are in `requirements.txt` (used by the Dockerfile).
- `cache_models.py` pre-caches models during the image build (also used by the Dockerfile).

## Credits
- Emilia Pipe inspiration and related ideas draw from OpenMMLab Amphion and community work.
import os
from yt_dlp import YoutubeDL
from typing import List

# --- Configuration ---
# Add all the YouTube channel URLs you want to process here.
CHANNEL_URLS: List[str] = [
    "https://www.youtube.com/@CodeAesthetic",
    # "https://www.youtube.com/@anotherchannel",
    # "https://www.youtube.com/@yetanotherchannel",
]

# The output file that will contain all video URLs.
OUTPUT_FILE: str = "master_url_list.txt"

def collect_video_urls():
    """
    Connects to YouTube channels and extracts all video URLs using yt-dlp's
    fast 'extract_flat' option. The URLs are saved to a text file.
    """
    all_video_urls = set()
    
    # yt-dlp options to only extract video URLs without downloading.
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'ignoreerrors': True,
    }

    print("🚀 Starting URL collection...")
    with YoutubeDL(ydl_opts) as ydl:
        for channel_url in CHANNEL_URLS:
            print(f"🔗 Extracting video URLs from: {channel_url}")
            try:
                info = ydl.extract_info(channel_url, download=False)
                if info and info.get("entries"):
                    # Extract the video URL for each entry in the channel's playlist.
                    urls = {f"https://www.youtube.com/watch?v={e['id']}" for e in info['entries'] if e and 'id' in e}
                    print(f"    Found {len(urls)} videos.")
                    all_video_urls.update(urls)
                else:
                    print(f"    No videos found or failed to extract for {channel_url}.")
            except Exception as e:
                print(f"[!] Error processing channel {channel_url}: {e}")

    # Write all collected URLs to the output file.
    with open(OUTPUT_FILE, "w") as f:
        for url in all_video_urls:
            f.write(f"{url}\n")
    
    print(f"\n✅ Success! Collected {len(all_video_urls)} unique video URLs.")
    print(f"Master list saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_video_urls()

import subprocess
import os
import random

# Setup - instal curl-cffi for impersonation and ffmpeg for auto converting

# --- Configuration ---
# Your Oxylabs proxy URL, built from your curl command.
# Format: http://<username>:<password>@<proxy_host>:<port>
# The username 'customer-leo3t_n7lBx-cc-US' tells Oxylabs to use a US-based IP.
# For session rotation, you can add session parameters to the username.
# e.g., customer-leo3t_n7lBx-cc-US-sessid-RANDOM
proxy_url = os.environ.get("PROXY_URL")

# The YouTube video you want to download for the test.
# Using a short, Creative Commons video is great for testing.
video_url = "https://www.youtube.com/watch?v=gA6r7iVzP6M" # "Big Buck Bunny"

# Where to save the downloaded audio and its format
output_folder = "audio_downloads"
output_template = os.path.join(output_folder, "%(title)s.%(ext)s")

# --- Create Output Folder ---
# Ensure the directory to save the file in exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

# --- Build and Run the Command ---
try:
    print(f"Attempting to download audio from: {video_url}")
    print(f"Using Oxylabs proxy server: pr.oxylabs.io:7777")
    print("Applying evasion techniques: Browser impersonation and sleep interval.")

    # This list contains the command and all its arguments.
    # This is the safest way to pass arguments to a subprocess.
    # Final, production-ready command list
    command = [
        "yt-dlp",

        # --- Proxy Configuration ---
        # "--proxy", proxy_url,

        # --- Format Selection (CRITICAL) ---
        # Request medium quality audio (96-128kbps range)
        # 1. M4A ~96-128kbps (to avoid transcoding)
        # 2. Any audio format ~96-128kbps
        # 3. Fallback to any M4A
        # 4. Last resort: any audio
        "-f", "bestaudio[ext=m4a][abr<=128]/bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio",

        # --- Evasion & Browser Impersonation ---
        "--impersonate", "chrome",                  # Mimic latest Chrome browser
        "--add-header", "Accept-Language: en-US,en;q=0.9",
        "--add-header", "DNT: 1",
        "--add-header", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "--add-header", "Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7",
        "--add-header", "Accept-Encoding: gzip, deflate, br",

        # --- Rate Limiting & Sleep ---
        "--limit-rate", "250K",                     # Throttle to 250 KB/s (reduced from 500K)
        "--sleep-interval", "2",                    # Sleep 2s between requests
        "--max-sleep-interval", "5",                # Max sleep of 5s if needed
        "--sleep-requests", "3",                    # Sleep after every 3 requests

        # --- Robustness & Error Handling ---
        "--retries", "5",                           # Retry failed downloads 5 times
        "--fragment-retries", "5",                  # Retry failed fragments 5 times
        "--ignore-errors",                          # Don't stop if one video fails
        "--no-playlist",                            # Don't download playlists by accident
        "--concurrent-fragments", "1",              # Download fragments sequentially

        # --- Audio Extraction ---
        "--extract-audio",                          # Extract audio
        "--audio-format", "m4a",                    # Force M4A output

        # --- Output & Logging ---
        # "--download-archive", "archive.txt",        # Track downloaded files
        "-o", output_template,                      # Set output filename template
        "--no-mtime",                               # Don't set file modification time
        "--quiet",                                  # Suppress normal output
        "--no-warnings",                            # Suppress warnings

        # --- Target ---
        video_url
    ]

    # Execute the command using subprocess
    # 'check=True' will raise an error if yt-dlp fails
    subprocess.run(command, check=True)

    print("\n✅ Download completed successfully!")
    print(f"Audio saved in the '{output_folder}' directory.")

except FileNotFoundError:
    print("\n❌ Error: 'yt-dlp' command not found.")
    print("Please ensure that yt-dlp is installed and accessible in your system's PATH.")
    print("You can install it with: pip install -U yt-dlp")
    print("For impersonation support: pip install --force-reinstall 'yt-dlp[impersonate]'")
    print("Also required: pip install requests-toolbelt")

except subprocess.CalledProcessError as e:
    print(f"\n❌ An error occurred while running yt-dlp.")
    print("This could be due to a proxy error, a network issue, or a problem with the video.")
    print("Check your Oxylabs dashboard for any connection issues.")
    print(f"yt-dlp exited with return code: {e.returncode}")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")

import subprocess
import os
import random

# --- Configuration ---
# Your Oxylabs proxy URL, built from your curl command.
# Format: http://<username>:<password>@<proxy_host>:<port>
# The username 'customer-leo3t_n7lBx-cc-US' tells Oxylabs to use a US-based IP.
# For session rotation, you can add session parameters to the username.
# e.g., customer-leo3t_n7lBx-cc-US-sessid-RANDOM
proxy_url = "http://customer-leo3t_n7lBx-cc-US:Weewoo_0242_@pr.oxylabs.io:7777"

# The YouTube video you want to download for the test.
# Using a short, Creative Commons video is great for testing.
video_url = "https://www.youtube.com/watch?v=hJnAHzo4-KI" # "Big Buck Bunny"

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
        "--proxy", proxy_url,

        # --- Evasion & Throttling Mitigation ---
        "--impersonate", "chrome",                  # Mimic latest Chrome browser
        "--add-header", "Accept-Language: en-US,en;q=0.9", # Standard browser header
        "--limit-rate", "500K",                     # Throttle download speed to 500 KB/s
        # "--sleep-interval", str(random.randint(30, 45)), # Wait 30-45s after each download

        # --- Format Selection (CRITICAL) ---
        # Prioritize a direct M4A stream copy. Avoids re-encoding.
        "-f", "bestaudio[ext=m4a]/bestaudio",

        # --- Robustness & Error Handling ---
        "--retries", "5",                           # Retry failed downloads 5 times
        "--fragment-retries", "5",                  # Retry failed fragments 5 times
        "--ignore-errors",                          # Don't stop if one video fails
        "--no-playlist",                            # Don't download playlists by accident

        # --- Output & Logging ---
        "--download-archive", "archive.txt",        # Track downloaded files
        "-o", output_template,                      # Set output filename template
        "--no-mtime",                               # Don't set file modification time
        # "--quiet",                                  # Suppress normal output
        # "--no-warnings",                            # Suppress warnings
        # "--print-json",                             # Print structured JSON info for logging

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

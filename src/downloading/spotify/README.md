# Spotify Podcast Downloader

Two scripts to download podcast episodes from the Spotify 100k dataset metadata onto S3.

### File Breakdown

`generate_tasks.py` takes in a JSONL of podcast episodes (through the `S3_BUCKET_NAME` and `S3_KEY_FOR_JSONL` variables near the top) and finds every episode in that dataset, enqueueing their episode URLs by writing `.task` files into S3 under the todo prefix.

`download_worker.py` consumes from an S3-based task queue, popping episode tasks and downloading the audio files to S3

### How to Run

0. Make sure the .jsonl file containing the Spotify 100k dataset metadata is in S3 (under /source-data/) 
1. Set env variables (AWS credentials) and S3 settings if needed
2. Set `S3_BUCKET_NAME` and `S3_KEY_FOR_JSONL` to your dataset location
3. Run `generate_tasks.py` to enqueue all episode URLs
4. Run `download_worker.py` to download all episode audio to S3

### Example AWS EC2 Setup

You only need one worker for generate_tasks, but the download_worker can be scaled to any amount of workers. I found 10 to be a good amount.

Below are the user data scripts (they run at first boot) for EC2 instances.

For `generate_tasks.py` on t2.micro
```
#!/bin/bash

# 1. Update system packages
sudo yum update -y

# 2. Install necessary software: Git and Python's package installer (pip)
sudo yum install -y git python3-pip

# 3. Navigate to the home directory of the default ec2-user
cd /home/ec2-user

# 4. Clone your public GitHub repository
git clone https://github.com/fixie-ai/audio-pipeline

# 5. Install the required Python library for AWS interaction
pip3 install boto3

# 6. Change the ownership of the cloned repository to the ec2-user
# This is important so you can easily edit files as the user, not root.
chown -R ec2-user:ec2-user audio-pipeline

python3 audio-pipeline/src/downloading/spotify/generate_tasks.py

shutdown -h now
```

For `download_worker.py` on c5.large
```
#!/bin/bash

# 1. Update system packages to the latest versions using dnf
sudo dnf update -y

# 2. Install necessary software (git and pip) using dnf
sudo dnf install -y git python3-pip

# 3. Install FFmpeg
#    Navigate to a temporary directory for the download
cd /home/ec2-user
#    Download the static build from the official source
curl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
#    Extract the archive
tar -xf ffmpeg-release-amd64-static.tar.xz
#    Move into the newly created directory
cd ffmpeg-*-amd64-static
#    Move the ffmpeg and ffprobe binaries to a location in the system's PATH
mv ffmpeg ffprobe /usr/local/bin/
#    Clean up the downloaded archive and extracted folder
cd ..
rm -rf ffmpeg-*-amd64-static ffmpeg-release-amd64-static.tar.xz

# 4. Navigate back to the ec2-user home directory
cd /home/ec2-user

# 5. Clone your public GitHub repository containing the worker scripts
git clone https://github.com/fixie-ai/audio-pipeline

# 6. Install the required Python library for AWS interaction
pip3 install boto3 requests

# 7. Change the ownership of the cloned repository to the ec2-user
#    This is important so you can easily run and edit files as the standard user.
chown -R ec2-user:ec2-user audio-pipeline

python3 -u audio-pipeline/src/downloading/spotify/download_worker.py

shutdown -h now

```
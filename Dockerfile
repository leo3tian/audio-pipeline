# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- System packages ---
RUN apt-get update -qq && \
    apt-get install -y -qq \
    git wget ffmpeg curl build-essential bzip2 libsndfile1 python3.9 python3-pip unzip && \
    rm -rf /var/lib/apt/lists/*

# --- Install AWS CLI v2 ---
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# --- Install Conda ---
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# --- Copy your entire project ---
WORKDIR /workspace
COPY . .

# --- Create Conda env and install dependencies ---
RUN /opt/conda/bin/conda create -n AudioPipeline python=3.9 -y && \
    # Step 1: Install heavy packages with Conda first
    /opt/conda/bin/conda run -n AudioPipeline conda install ffmpeg pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    # Step 2: Install all other packages with pip from the single requirements file
    /opt/conda/bin/conda run -n AudioPipeline pip install --no-cache-dir -r requirements.txt

# --- Pre-cache models to prevent race conditions ---
RUN /opt/conda/envs/AudioPipeline/bin/python cache_models.py

# --- Create a persistent entrypoint script ---
# This script activates the conda environment and then executes any command passed to it.
RUN echo '#!/bin/bash' > /workspace/entrypoint.sh && \
    echo 'set -e' >> /workspace/entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /workspace/entrypoint.sh && \
    echo 'conda activate AudioPipeline' >> /workspace/entrypoint.sh && \
    echo 'exec "$@"' >> /workspace/entrypoint.sh && \
    chmod +x /workspace/entrypoint.sh

# --- Set working dir & entrypoint ---
WORKDIR /workspace
ENTRYPOINT ["/workspace/entrypoint.sh"]

# The default command if none is provided to `docker run` (e.g., for interactive mode)
CMD ["/bin/bash"]

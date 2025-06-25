# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- System packages ---
RUN apt update && apt install -y \
    git wget ffmpeg curl build-essential bzip2 libsndfile1 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# --- Install Conda ---
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh
#RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
#    bash Miniforge3-Linux-aarch64.sh -b -p /opt/conda && \
#    rm Miniforge3-Linux-aarch64.sh
ENV PATH="/opt/conda/bin:$PATH"

# --- Copy your entire project ---
WORKDIR /workspace
COPY . .

# --- Create and set up Conda env for Emilia and downloader ---
RUN /opt/conda/bin/conda create -n AudioPipeline python=3.9 -y && \
    /opt/conda/bin/conda run -n AudioPipeline pip install yt-dlp && \
    /opt/conda/bin/conda run -n AudioPipeline bash -c "cd Emilia && bash env.sh"

# --- Set working dir & launch shell ---
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-l"]
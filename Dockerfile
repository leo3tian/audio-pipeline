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
ENV PATH="/opt/conda/bin:$PATH"

# --- Copy your entire project ---
WORKDIR /workspace
COPY . .

# --- Create Conda env and install dependencies ---
# This now includes huggingface_hub for the uploader
RUN /opt/conda/bin/conda create -n AudioPipeline python=3.9 -y && \
    /opt/conda/bin/conda run -n AudioPipeline pip install yt-dlp huggingface_hub datasets && \
    /opt/conda/bin/conda run -n AudioPipeline bash -c "cd Emilia && bash env.sh"

# --- RECOMMENDED CHANGE: Pre-cache models to prevent race conditions ---
# This runs our new script inside the conda environment to download models
# into the Docker image layer itself.
RUN /opt/conda/envs/AudioPipeline/bin/python cache_models.py

# --- Auto-activate Conda environment on login ---
# By adding these commands to /root/.profile, they will be executed
# automatically every time a login shell starts.
RUN { \
        echo; \
        echo '# Activate AudioPipeline Conda environment'; \
        echo 'source /opt/conda/etc/profile.d/conda.sh'; \
        echo 'conda activate AudioPipeline'; \
    } >> /root/.profile

# --- Set working dir & default command ---
WORKDIR /workspace
# Set the default command to be a bash shell for interactive use.
# For automated runs, this can be overridden.
CMD ["/bin/bash", "-l"]

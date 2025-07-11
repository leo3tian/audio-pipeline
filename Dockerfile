# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- System packages ---
# Using apt-get which is better for scripting.
# We'll install most packages, and handle aws-cli separately.
RUN apt-get update -qq && \
    apt-get install -y -qq \
    git wget ffmpeg curl build-essential bzip2 libsndfile1 python3 python3-pip unzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install boto3

# --- Install AWS CLI v2 ---
# Using the official AWS installer is more reliable than apt-get in some base images.
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
# Added boto3 for AWS S3 access
RUN /opt/conda/bin/conda create -n AudioPipeline python=3.9 -y && \
    /opt/conda/bin/conda run -n AudioPipeline pip install yt-dlp huggingface_hub datasets boto3 soundfile && \
    /opt/conda/bin/conda run -n AudioPipeline bash -c "cd Emilia && bash env.sh"

# --- Pre-cache models to prevent race conditions ---
RUN /opt/conda/envs/AudioPipeline/bin/python cache_models.py

# --- Auto-activate Conda environment on login ---
RUN { \
        echo; \
        echo '# Activate AudioPipeline Conda environment'; \
        echo 'source /opt/conda/etc/profile.d/conda.sh'; \
        echo 'conda activate AudioPipeline'; \
    } >> /root/.profile

# --- Set working dir & default command ---
WORKDIR /workspace
CMD ["/bin/bash", "-l"]

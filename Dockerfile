# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System packages (screen & vim for running and debugging)
RUN apt-get update -qq && \
    apt-get install -y -qq \
    git wget ffmpeg curl build-essential bzip2 libsndfile1 python3 python3-pip unzip \
    libsox-dev screen vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Boto3 used to interact with S3 and R2
RUN pip install boto3

# Install AWS CLI (used to upload to S3)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Install conda and set path
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Copy workspace
WORKDIR /workspace

# Copy requirements and install all conda envs
COPY requirements.txt .
COPY Emilia/env.sh Emilia/env.sh
RUN /opt/conda/bin/conda create -n AudioPipeline python=3.9 -y && \
    /opt/conda/bin/conda install -n AudioPipeline -y ffmpeg && \
    /opt/conda/bin/conda install -n AudioPipeline -c pytorch -c nvidia -y \
        pytorch torchvision torchaudio pytorch-cuda=12.4 && \
    /opt/conda/bin/conda run -n AudioPipeline pip install -r requirements.txt && \
    /opt/conda/bin/conda clean -a -y

# Ensure onnxruntime GPU wheel is installed (and not the CPU wheel), then verify CUDA EP
RUN /opt/conda/bin/conda run -n AudioPipeline pip uninstall -y onnxruntime onnxruntime-gpu && \
    /opt/conda/bin/conda run -n AudioPipeline pip install --no-cache-dir onnxruntime-gpu==1.19.2 && \
    /opt/conda/bin/conda run -n AudioPipeline python -c "import onnxruntime as ort; provs=ort.get_available_providers(); print('onnxruntime providers:', provs); assert 'CUDAExecutionProvider' in provs, provs"

# Now copy the rest of the workspace
COPY . .

# Cache models
RUN /opt/conda/bin/conda run -n AudioPipeline python cache_models.py

# Activate conda env
RUN { \
        echo; \
        echo '# Activate AudioPipeline Conda environment'; \
        echo 'source /opt/conda/etc/profile.d/conda.sh'; \
        echo 'conda activate AudioPipeline'; \
    } >> /root/.profile

# --- Set working dir & default command ---
WORKDIR /workspace
CMD ["/bin/bash", "-l"]

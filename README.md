### Setup

In the root directory of this project, build the docker container:
`docker build -t emilia-pipeline .`

Then run the docker container:
`docker run --gpus all -v $(pwd):/workspace -w /workspace -it emilia-pipeline`

Once in the docker container, run:
`source /opt/conda/etc/profile.d/conda.sh && conda activate AudioPipeline`

Then you are ready to run:
`python download.py`
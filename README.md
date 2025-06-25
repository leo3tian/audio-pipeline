### Credits
Credits to Amphion team: https://github.com/open-mmlab/Amphion
Uses Emilia-Pipe with slight modifications:
- Removed DNSMOS filter
- Changed output location
- Disabled logger output (except for errors)

### Setup
First download sig_bak_ovr.onnx and UVR-MDX-NET-Inst_HQ_3.onnx into yt-audio-pipeline/Emilia/models

In the root directory of this project, build the docker container:
`docker build -t emilia-pipeline .`

Then run the docker container:
`docker run --gpus all -v $(pwd):/workspace -w /workspace -it emilia-pipeline`

Once in the docker container, run:
`source /opt/conda/etc/profile.d/conda.sh && conda activate AudioPipeline`

Pick what youtube channels to download by editing CHANNEL_URLS global variable in download.py - ensure compliance with video copyright

Then you are ready to run:
`python download.py`
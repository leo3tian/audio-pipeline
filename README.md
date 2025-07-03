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
`docker run -it --env-file .env --gpus all -v $(pwd):/workspace -w /workspace -it emilia-pipeline`

Pick what youtube channels to download by editing CHANNEL_URLS in `0_get_urls.py` - ensure compliance with video copyright


# cache_models.py
# This script is run once during the Docker build process to pre-download
# and cache models from Torch Hub, preventing race conditions at runtime.

import torch

print("--- Caching models from Torch Hub ---")

try:
    print("Caching Silero VAD model...")
    # This command downloads the model to the torch hub cache inside the image.
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)
    print("✅ Silero VAD model cached successfully.")

    # If you had other models from torch.hub, you would add them here.
    # e.g. torch.hub.load(...)

except Exception as e:
    print(f"🔥 Failed to cache models: {e}")
    # Exit with a non-zero status code to fail the Docker build if caching fails
    exit(1)

print("--- Model caching complete ---")

# cache_models.py
# This script is run once during the Docker build process to pre-download
# and cache models from Torch Hub, preventing race conditions at runtime.

import torch

print("--- Caching models from Torch Hub ---")

try:
    print("Caching Silero VAD model...")
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True, trust_repo=True)
    print("✅ Silero VAD model cached successfully.")

except Exception as e:
    print(f"🔥 Failed to cache models: {e}")
    exit(1)

print("--- Model caching complete ---")

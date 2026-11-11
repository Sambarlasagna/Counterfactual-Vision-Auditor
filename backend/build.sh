#!/usr/bin/env bash
set -e

# Install CPU-only PyTorch first (much smaller, no CUDA libs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch/torchvision since already installed)
pip install -r requirements.txt --no-deps
pip install fastapi uvicorn pydantic pydantic-settings python-multipart \
    scikit-learn pillow numpy scipy requests


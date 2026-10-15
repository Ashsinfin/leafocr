#!/usr/bin/env bash
# Runs LeafOCR on the NVIDIA GPU via WSL2 (native-Windows TensorFlow is CPU-only).
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
exec ~/venvs/leafocr/bin/python app.py

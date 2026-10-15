#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
uv venv --allow-existing -q -p 3.11 ~/venvs/leafocr
UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 VIRTUAL_ENV=~/venvs/leafocr uv pip install -q "tensorflow[and-cuda]==2.19.*" opencv-python-headless scipy gradio
~/venvs/leafocr/bin/python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))" 2>&1 | tail -2

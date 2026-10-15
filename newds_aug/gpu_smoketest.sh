#!/usr/bin/env bash
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONIOENCODING=utf-8
~/venvs/leafocr/bin/python gpu_smoketest.py "$@" 2>&1 | grep -v -i "warning\|oneDNN\|rebuild TensorFlow\|cuda_\|computation_placer"

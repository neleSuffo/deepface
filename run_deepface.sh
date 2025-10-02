#!/bin/bash

INPUT_DIR="$1"
# CUDA library setup script for TensorFlow
# This script sets up the necessary environment variables before running Python

# Set up CUDA library paths
CUDA_LIB_PATHS=(
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cublas/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cudnn/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cufft/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/curand/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cusolver/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cusparse/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_runtime/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_cupti/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/nvjitlink/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/nvtx/lib"
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/nccl/lib"
)

# Filter existing paths and join them
EXISTING_PATHS=""
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -n "$EXISTING_PATHS" ]; then
            EXISTING_PATHS="${EXISTING_PATHS}:${path}"
        else
            EXISTING_PATHS="$path"
        fi
    fi
done

# Set environment variables
export LD_LIBRARY_PATH="${EXISTING_PATHS}:${LD_LIBRARY_PATH}"
export CUDA_MODULE_LOADING="LAZY"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Preload cuBLAS libraries
CUBLAS_LIB_PATH="/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cublas/lib"
export LD_PRELOAD="${CUBLAS_LIB_PATH}/libcublasLt.so.12:${CUBLAS_LIB_PATH}/libcublas.so.12"

echo "🔧 CUDA environment configured"

# Activate virtual environment and run the evaluation
cd /home/nele_pauline_suffo/projects/deepface
#python run_deepface.py
python run_face_clustering.py --input_dir "$INPUT_DIR"
#!/bin/bash

# Exit if a command fails
set -e

echo " Preparing HPC to Local Sync..."

# Load the .env file if it exists
if [ -f .env ]; then
    # Exporting variables, exclude comments
    export $(grep -v '^#' .env | xargs)
    echo " -> Loaded credentials from .env file."
else
    exit
fi

LOCAL_DIR="./AudioProcessing/"

echo " Pulling from: ${HPC_USER}@${HPC_HOST}:${HPC_PATH}"
echo "======================================"

# Rsync
rsync -avz -e "ssh -p ${HPC_PORT}" \
    --exclude 'data/' \
    --exclude '*.pth' \
    --exclude '*.nsys-rep' \
    --exclude '*.ncu-rep' \
    --exclude '__pycache__/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '.env' \
    "${HPC_USER}@${HPC_HOST}:${HPC_PATH}" "${LOCAL_DIR}"

echo " Sync Complete..."
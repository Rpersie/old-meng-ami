#!/bin/bash

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

# Train a convolutional variational multidecoder
echo "Training convolutional variational multidecoder..."
python3 cnn/scripts/train_vae.py
echo "Trained convolutional variational multidecoder."

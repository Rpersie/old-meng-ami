#!/bin/bash

. ./path.sh
. ./dnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

# Train a deep multidecoder
echo "Training deep multidecoder..."
python3 dnn/scripts/train_ae.py
echo "Trained deep multidecoder."

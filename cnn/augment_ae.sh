#!/bin/bash

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

# Perform data augmentation with a convolutional multidecoder
echo "Augmenting data using convolutional multidecoder..."
python3 cnn/scripts/augment_ae.py
echo "Done augmenting data using convolutional multidecoder."

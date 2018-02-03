#!/bin/bash

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
source activate env-cpu
echo "Environment set up."

mode=$1

python3 cnn/scripts/param_count.py $mode

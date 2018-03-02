#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=72:00:00
#SBATCH -J train_baseline
#SBATCH --exclude=sls-sm-[5],sls-tesla-[0,1]

echo "STARTING BASELINE ACOUSTIC MODEL TRAINING JOB"

. ./path.sh
. $MENG_ROOT/am/baseline_config.sh
. $MENG_ROOT/am/path-cuda.sh

if [ "$#" -lt 1 ]; then
    echo "Train domain not specified; exiting"
    exit 1
fi

train_domain=$1
echo "Train domain ${train_domain}"

expt_name="train_${train_domain}/baseline/${ARCH_NAME}"
mkdir -p $LOG_DIR/$expt_name

model_dir=$MODEL_DIR/$expt_name
mkdir -p $model_dir

for epoch in $(seq $START_EPOCH $END_EPOCH); do
    echo "========== EPOCH $epoch =========="

    step_size=$(echo "scale=10; 0.05 * 0.75 ^ ($epoch - 1)" | bc)
    echo "Step size now $step_size"

    epoch_log=$LOG_DIR/$expt_name/train_baseline-epoch${epoch}.log

    if [ ! -f $model_dir/param-$((epoch-1)) ]; then
        # Parameter file doesn't exist -- only generate if we're just starting
        if [ "$epoch" -eq "1" ]; then
            echo "TDNN not initialized. Initializing parameters..."
            $MENG_ROOT/am/init-tdnn.py random > $model_dir/param-$((epoch-1))
            echo "Done initializing parameters."
        else
            echo "Parameter file does not exist for (epoch - 1 = $((epoch-1)))"
            exit 1
        fi
    fi

    # Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
    OMP_NUM_THREADS=1 /data/sls/scratch/haotang/ami/dist/nn-20171213-4c6c341/nnbin/frame-tdnn-learn-gpu \
        --frame-scp $DATASET/${train_domain}-train-norm.blogmel.scp \
        --label-scp $DATASET/ihm-train-tri3.bali.scp \
        --param $model_dir/param-$((epoch-1)) \
        --opt-data $model_dir/opt-data-$((epoch-1)) \
        --output-param $model_dir/param-$epoch \
        --output-opt-data $model_dir/opt-data-$epoch \
        --label $DATASET/ihm-pdfids.txt \
        --seed $epoch \
        --shuffle \
        --opt const-step \
        --step-size $step_size \
        --clip 5 \
        > $epoch_log

    # Show average E at end to make sure training progresses correctly
    $MENG_ROOT/am/avg-e.py 100 < $epoch_log
done

echo "DONE BASELINE ACOUSTIC MODEL TRAINING JOB"

#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_baseline
#SBATCH --exclude=sls-sm-[1,2,4]

echo "STARTING BASELINE ACOUSTIC MODEL TRAINING JOB"

. ./path.sh
. $MENG_ROOT/am/job_config.sh
. $MENG_ROOT/am/path-cuda.sh

mkdir -p $LOGS/$EXPT_NAME
train_log=$LOGS/$EXPT_NAME/train_baseline.log

if [ ! -f $MODEL_DIR/param-$((EPOCH-1)) ]; then
    # Parameter file doesn't exist -- only generate if we're just starting
    if [ "$EPOCH" -eq "1" ]; then
        echo "TDNN not initialized. Initializing parameters..."
        $MENG_ROOT/am/init-tdnn.py random > $MODEL_DIR/param-$((EPOCH-1))
        echo "Done initializing parameters."
    else
        echo "Parameter file does not exist for (epoch - 1 = $((EPOCH-1)))"
        exit 1
    fi
fi

echo "Training TDNN..."
OMP_NUM_THREADS=1 /data/sls/scratch/haotang/ami/dist/nn-20171210-5b69f7f/nnbin/frame-tdnn-learn-gpu \
    --frame-scp $DATASET/${DOMAIN}-train-norm.blogmel.scp \
    --label-scp $DATASET/${DOMAIN}-train-tri3.bali.scp \
    --param $MODEL_DIR/param-$((EPOCH-1)) \
    --opt-data $MODEL_DIR/opt-data-$((EPOCH-1)) \
    --output-param $MODEL_DIR/param-$EPOCH \
    --output-opt-data $MODEL_DIR/opt-data-$EPOCH \
    --label $DATASET/${DOMAIN}-pdfids.txt \
    --seed $EPOCH \
    --shuffle \
    --opt const-step \
    --step-size 0.05 \
    --clip 5 \
    > $train_log
echo "Trained TDNN."

echo "DONE BASELINE ACOUSTIC MODEL TRAINING JOB"

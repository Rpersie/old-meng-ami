export EPOCH=0
# export DATASET=/data/sls/scratch/haotang/ami/sls-data/ami-0.1
export DATASET=$TEST_FEATS
export DOMAIN=ihm

export EXPT_NAME="${DATASET}/${DOMAIN}/frame-tdnn-450x7-step0.05"

export MODEL_DIR=$MODELS/am/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/am/logs
mkdir -p $LOGS

export EPOCH=1
export DEBUG_MODEL=true
if [ "$DEBUG_MODEL" = true ] ; then
    export DATASET_NAME=debug
    export DATASET=$TEST_FEATS
else
    export DATASET_NAME=ami-0.1
    export DATASET=/data/sls/scratch/haotang/ami/sls-data/${DATASET_NAME}
fi
export DOMAIN=ihm

export EXPT_NAME="${DATASET_NAME}/${DOMAIN}/frame-tdnn-450x7-step0.05"

export MODEL_DIR=$MODELS/am/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/am/logs
mkdir -p $LOGS

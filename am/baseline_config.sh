# For training
export START_EPOCH=1
export END_EPOCH=15

# For prediction
export MODEL_EPOCH=15

export DEBUG_MODEL=false
if [ "$DEBUG_MODEL" = true ] ; then
    export DATASET_NAME=debug
    export DATASET=$TEST_FEATS
else
    # export DATASET_NAME=ami-0.1
    export DATASET_NAME=ami-full
    export DATASET=$MENG_ROOT/${DATASET_NAME}
fi

export GOLD_DIR=/data/sls/scratch/haotang/ami/sls-data/${DATASET_NAME}

# Always use IHM pdfids! (See Hao email from 1/17/18)
export NPRED=3984

export ARCH_NAME="frame-tdnn-450x7-step0.05"


export MODEL_DIR=$MODELS/am/$DATASET_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/am/$DATASET_NAME
mkdir -p $LOG_DIR

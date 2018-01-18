# For training
export START_EPOCH=1
export END_EPOCH=20

# For prediction
export MODEL_EPOCH=20

export DATASET_NAME=ami-0.1
export DATASET=$MENG_ROOT/${DATASET_NAME}

export CNN_NAME=ENC_C_256_256_128_K_3_3_3_P_3_3_0_F_1024_1024/LATENT_1024/DEC_F_1024_1024_C_128_256_256_K_3_3_3_P_0_3_3/ACT_SELU/OPT_Adam_LR_0.0001_EPOCHS_100_BATCH_256_DEBUG_false/ae
export AUGMENTED_DATA_DIR=${SCRATCH}/augmented_data/cnn/$CNN_NAME

export SOURCE_DOMAIN=sdm1
export TARGET_DOMAIN=ihm
export PREDICT_DOMAIN=ihm
export GOLD_DIR=/data/sls/scratch/haotang/ami/sls-data/${DATASET_NAME}

# if [ "$TRAIN_DOMAIN" = ihm ] ; then
#     export NPRED=3984
# else
#     # SDM1
#     export NPRED=3966
# fi

# Always use IHM pdfids! (See Hao email from 1/17/18)
export NPRED=3984

export EXPT_NAME="${DATASET_NAME}/train_${TARGET_DOMAIN}/augmented_src_${SOURCE_DOMAIN}/frame-tdnn-450x7-step0.05/${CNN_NAME}"

export MODEL_DIR=$MODELS/am/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/am/logs
mkdir -p $LOGS

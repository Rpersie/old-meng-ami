# For training
export START_EPOCH=1
export END_EPOCH=20

# For prediction
export MODEL_EPOCH=20

export DATASET_NAME=feats_ami-0.1
export DATASET=$MENG_ROOT/${DATASET_NAME}

export CNN_NAME=ENC_C_256_256_128_K_3_3_3_P_3_3_0_F_1024_1024/LATENT_1024/DEC_F_1024_1024_C_128_256_256_K_3_3_3_P_0_3_3/ACT_SELU/OPT_Adam_LR_0.00001_EPOCHS_100_BATCH_256_DEBUG_false
export AUGMENTED_DATA_DIR=${SCRATCH}/augmented_data/cnn/$CNN_NAME

export EXPT_NAME="${DATASET_NAME}/sdm1_sim/frame-tdnn-450x7-step0.05"

export MODEL_DIR=$MODELS/am/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/am/logs
mkdir -p $LOGS

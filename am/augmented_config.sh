# For training
export START_EPOCH=8
export END_EPOCH=15

# For prediction
export MODEL_EPOCH=15

export NOISE_RATIO=0.0
# export DATASET_NAME=ami-0.1
export DATASET_NAME=ami-full
export DATASET=$MENG_ROOT/${DATASET_NAME}

# For adversarial multidecoders
export DOMAIN_ADV_FC=( 512 512 )
export DOMAIN_ADV_ACTIVATION=Sigmoid
export DOMAIN_ADV_FC_DELIM=$(printf "_%s" "${DOMAIN_ADV_FC[@]}")

# For generative adversarial multidecoders
export GAN_FC=( 512 512 )
export GAN_ACTIVATION=Sigmoid
export GAN_FC_DELIM=$(printf "_%s" "${GAN_FC[@]}")

export CNN_NAME=ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false
export AUGMENTED_DATA_BASE_DIR=${AUGMENTED_DATA}/cnn/$DATASET_NAME/$CNN_NAME

export GOLD_DIR=/data/sls/scratch/haotang/ami/sls-data/${DATASET_NAME}

# Always use IHM pdfids! (See Hao email from 1/17/18)
export NPRED=3984

# export ARCH_NAME="frame-tdnn-450x7-step0.05"
export ARCH_NAME="frame-tdnn-450x7-step0.05-decay"

export MODEL_DIR=$MODELS/am/$DATASET_NAME
mkdir -p $MODEL_DIR

export LOG_DIR=${LOGS}/am/$DATASET_NAME
mkdir -p $LOG_DIR

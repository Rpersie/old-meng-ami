# For training
export START_EPOCH=1
export END_EPOCH=10

# For prediction
export MODEL_EPOCH=10

# export DATASET_NAME=ami-0.1
export DATASET_NAME=ami-full
export DATASET=$MENG_ROOT/${DATASET_NAME}

export MODEL_TYPE=ae
export NOISE_RATIO=0.0

# For adversarial multidecoders
export DOMAIN_ADV_FC=( 512 512 )
export DOMAIN_ADV_ACTIVATION=Sigmoid
export DOMAIN_ADV_FC_DELIM=$(printf "_%s" "${DOMAIN_ADV_FC[@]}")

# For generative adversarial multidecoders
export GAN_FC=( 512 512 )
export GAN_ACTIVATION=Sigmoid
export GAN_FC_DELIM=$(printf "_%s" "${GAN_FC[@]}")

export CNN_NAME=ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false
export AUGMENTED_DATA_BASE_DIR=${SCRATCH}/augmented_data/cnn/$DATASET_NAME/$CNN_NAME

export PREDICT_DOMAIN=sdm1
export GOLD_DIR=/data/sls/scratch/haotang/ami/sls-data/${DATASET_NAME}

# if [ "$TRAIN_DOMAIN" = ihm ] ; then
#     export NPRED=3984
# else
#     # SDM1
#     export NPRED=3966
# fi

# Always use IHM pdfids! (See Hao email from 1/17/18)
export NPRED=3984

export EXPT_NAME="combined/frame-tdnn-450x7-step0.05/${CNN_NAME}"

export MODEL_DIR=$MODELS/am/$DATASET_NAME/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/am/logs/$DATASET_NAME
mkdir -p $LOGS

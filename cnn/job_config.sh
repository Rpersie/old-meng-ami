export FEAT_DIM=80      # 80 log-Mel
export LEFT_CONTEXT=5
export RIGHT_CONTEXT=5

export OPTIMIZER=Adam
export LEARNING_RATE=0.0001
export EPOCHS=100
export BATCH_SIZE=256

export ENC_CHANNELS=( 32 )
export ENC_KERNELS=( 5 )
export ENC_POOLS=( 5 )          # Use 0 to indicate no pooling
export LATENT_DIM=1024
export DEC_CHANNELS=( 32 )
export DEC_KERNELS=( 5 )
export DEC_POOLS=( 5 )          # Use 0 to indicate no pooling
export ACTIVATION_FUNC=SELU

export ENC_CHANNELS_DELIM=$(printf "_%s" "${ENC_CHANNELS[@]}")
export ENC_KERNELS_DELIM=$(printf "_%s" "${ENC_KERNELS[@]}")
export ENC_POOLS_DELIM=$(printf "_%s" "${ENC_POOLS[@]}")
export DEC_CHANNELS_DELIM=$(printf "_%s" "${DEC_CHANNELS[@]}")
export DEC_KERNELS_DELIM=$(printf "_%s" "${DEC_KERNELS[@]}")
export DEC_POOLS_DELIM=$(printf "_%s" "${DEC_POOLS[@]}")

export DECODER_CLASSES=( ihm sdm1 )
export DECODER_CLASSES_DELIM=$(printf "_%s" "${DECODER_CLASSES[@]}")

export DEBUG_MODEL=false
if [ "$DEBUG_MODEL" = true ] ; then
    export CURRENT_FEATS=$TEST_FEATS
else
    export CURRENT_FEATS=$FEATS
fi

export EXPT_NAME="ENC_${ENC_CHANNELS_DELIM}_KERN_${ENC_KERNELS_DELIM}_POOL_${ENC_POOLS_DELIM}/LATENT_${LATENT_DIM}/DEC_${DEC_CHANNELS_DELIM}_KERN_${DEC_KERNELS_DELIM}_POOL_${DEC_POOLS_DELIM}/ACT_${ACTIVATION_FUNC}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}_DEBUG_${DEBUG_MODEL}"

export MODEL_DIR=${MODELS}/cnn/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/cnn/logs
mkdir -p $LOGS

export FEAT_DIM=80      # 80 log-Mel
export LEFT_CONTEXT=5
export RIGHT_CONTEXT=5

export OPTIMIZER=Adam
export LEARNING_RATE=0.001
export EPOCHS=50
export BATCH_SIZE=256

export ENC_LAYERS=( )
export LATENT_DIM=512
export DEC_LAYERS=( )
export ACTIVATION_FUNC=SELU

export ENC_LAYERS_DELIM=$(printf "_%s" "${ENC_LAYERS[@]}")
export DEC_LAYERS_DELIM=$(printf "_%s" "${DEC_LAYERS[@]}")

export DECODER_CLASSES=( ihm sdm1 )
export DECODER_CLASSES_DELIM=$(printf "_%s" "${DECODER_CLASSES[@]}")

export DEBUG_MODEL=true
if [ "$DEBUG_MODEL" = true ] ; then
    export CURRENT_FEATS=$TEST_FEATS
else
    export CURRENT_FEATS=$FEATS
fi

export EXPT_NAME="ENC_${ENC_LAYERS_DELIM}_LATENT_${LATENT_DIM}_DEC_${DEC_LAYERS_DELIM}_ACT_${ACTIVATION_FUNC}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}_DEBUG_${DEBUG_MODEL}"

# For decoding
export CURRENT_DECODER_CLASS=ihm

export MODEL_DIR=$MODELS/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${PWD}/logs
mkdir -p $LOGS

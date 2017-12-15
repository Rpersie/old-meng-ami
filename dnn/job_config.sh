export FEAT_DIM=80      # 80 log-Mel
export LEFT_SPLICE=5
export RIGHT_SPLICE=5

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

export MAIN_DECODER_CLASS=ihm
export DECODER_CLASSES=( ihm sdm1 )
export DECODER_CLASSES_DELIM=$(printf "_%s" "${DECODER_CLASSES[@]}")
export TRAIN_DIRS=( )
export DEV_DIRS=( )
export EVAL_DIRS=( )

debug_model=false

for ((i=0;i<${#DECODER_CLASSES[@]};i++))
do
    decoder_class="${DECODER_CLASSES[i]}"

    if [ "$debug_model" = true ] ; then
        TRAIN_DIRS[$i]="${FEATS}/${decoder_class}/train-logmel-hires/data"
        DEV_DIRS[$i]="${FEATS}/${decoder_class}/dev-logmel-hires/data"
        EVAL_DIRS[$i]="${FEATS}/${decoder_class}/eval-logmel-hires/data"
    else
        TRAIN_DIRS[$i]="${TEST_FEATS}/${decoder_class}/train"
        DEV_DIRS[$i]="${TEST_FEATS}/${decoder_class}/dev"
        EVAL_DIRS[$i]="${TEST_FEATS}/${decoder_class}/eval"
    fi
done
export TRAIN_DIRS_DELIM=$(printf " %s" "${TRAIN_DIRS[@]}")
export DEV_DIRS_DELIM=$(printf " %s" "${DEV_DIRS[@]}")
export EVAL_DIRS_DELIM=$(printf " %s" "${EVAL_DIRS[@]}")

export EXPT_NAME="MAIN_${MAIN_DECODER_CLASS}/ENC_${ENC_LAYERS_DELIM}_LATENT_${LATENT_DIM}_DEC_${DEC_LAYERS_DELIM}_ACT_${ACTIVATION_FUNC}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}"

# For decoding
export CURRENT_DECODER_CLASS=ihm

export MODEL_DIR=$MODELS/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${PWD}/logs
mkdir -p $LOGS

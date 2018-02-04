export FEAT_DIM=80      # 80 log-Mel
export LEFT_CONTEXT=5
export RIGHT_CONTEXT=5

export OPTIMIZER=Adam
export LEARNING_RATE=0.0001
export EPOCHS=2
export BATCH_SIZE=256

export ENC_CHANNELS=( 256 256 64 )
export ENC_KERNELS=( 5 3 3 )        # Assume square kernels (AxA)
export ENC_POOLS=( 2 2 2 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling
export ENC_FC=( 2048 )     # Fully-connected layers following conv layers

export LATENT_DIM=1024

export DEC_FC=( 2048 )     # Fully-connected layers before conv layers
export DEC_CHANNELS=( 64 256 256 )
export DEC_KERNELS=( 3 3 5 )        # Assume square kernels (AxA)
export DEC_POOLS=( 2 2 2 )          # Pool only in frequency; no overlap. Use 0 to indicate no pooling

export USE_BATCH_NORM=false
export ACTIVATION_FUNC=SELU
export WEIGHT_INIT=xavier_uniform

export ENC_CHANNELS_DELIM=$(printf "_%s" "${ENC_CHANNELS[@]}")
export ENC_KERNELS_DELIM=$(printf "_%s" "${ENC_KERNELS[@]}")
export ENC_POOLS_DELIM=$(printf "_%s" "${ENC_POOLS[@]}")
export ENC_FC_DELIM=$(printf "_%s" "${ENC_FC[@]}")

export DEC_FC_DELIM=$(printf "_%s" "${DEC_FC[@]}")
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
export PROFILE_RUN=true

export EXPT_NAME="ENC_C${ENC_CHANNELS_DELIM}_K${ENC_KERNELS_DELIM}_P${ENC_POOLS_DELIM}_F${ENC_FC_DELIM}/LATENT_${LATENT_DIM}/DEC_F${DEC_FC_DELIM}_C${DEC_CHANNELS_DELIM}_K${DEC_KERNELS_DELIM}_P${DEC_POOLS_DELIM}/ACT_${ACTIVATION_FUNC}_BN_${USE_BATCH_NORM}_WEIGHT_INIT_${WEIGHT_INIT}/OPT_${OPTIMIZER}_LR_${LEARNING_RATE}_EPOCHS_${EPOCHS}_BATCH_${BATCH_SIZE}_DEBUG_${DEBUG_MODEL}"

export MODEL_DIR=${MODELS}/cnn/$EXPT_NAME
mkdir -p $MODEL_DIR

export LOGS=${MENG_ROOT}/cnn/logs
mkdir -p $LOGS

# For data augmentation
export AUGMENTED_DATA_DIR=${SCRATCH}/augmented_data/cnn/$EXPT_NAME
mkdir -p $AUGMENTED_DATA_DIR

# For viewing activations on dev set
export TOP_COUNT=100    # Only the average of images with top responses logged
export ACTIVATIONS_DIR=${SCRATCH}/activations/cnn/top_${TOP_COUNT}/$EXPT_NAME
mkdir -p $ACTIVATIONS_DIR

# Denoising autoencoder parameters; uses input "destruction" as described in
# "Extracting and Composing Robust Features with Denoising Autoencoders", Vincent et. al.
# http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/176
# Basically sets (NOISE_RATIO * 100)% of input features to 0 at random
export NOISE_RATIO=0.25

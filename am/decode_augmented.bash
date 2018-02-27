#!/bin/bash
#SBATCH -p 630
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --array=1-30
#SBATCH -J decode_augmented

echo "STARTING AUGMENTED ACOUSTIC MODEL DECODING JOB"

. ./path.sh
. $MENG_ROOT/am/augmented_config.sh
. $MENG_ROOT/am/path-opt.sh

if [ "$#" -lt 4 ]; then
    echo "Run mode, source domain, target domain and predict domain not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

src_domain=$2
tar_domain=$3
predict_domain=$4
echo "Source domain ${src_domain}, target domain ${tar_domain}, predict domain ${predict_domain}"

domain_adversarial=false
gan=false
if [ "$#" -ge 4 ]; then
    if [ "$5" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi
    
    if [ "$5" == "gan" ]; then
        gan=true
        echo "Using generative adversarial net (GAN) style training"
    fi
fi

split=$SLURM_ARRAY_TASK_ID

expt_name="train_${tar_domain}/augmented_src_${src_domain}/${ARCH_NAME}/${CNN_NAME}"

mkdir -p $LOGS/$expt_name

if [ "$domain_adversarial" == true ]; then
    log_dir=$LOGS/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    log_dir=$LOGS/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    log_dir=$LOGS/$expt_name/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/$expt_name/${run_mode}_ratio${NOISE_RATIO}
fi

decode_dir=$log_dir/decode_${predict_domain}
mkdir -p $decode_dir
decode_log=$decode_dir/decode_${split}.log

if [ "$DATASET" == "ami-full" ]; then
    # Didn't bother copying all of these SCPs over
    frame_split_dir=$AMI/split30/${predict_domain}-dev-norm-$((split - 1)).blogmel.scp
else
    frame_split_dir=$DATASET/split30/${predict_domain}-dev-norm-$((split - 1)).blogmel.scp
fi

echo "Predicting log probabilities for split ${split}..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
# Only use these environment variables if on 630 or 520 machines -- 510s don't work with them!
# OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
/data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $frame_split_dir/${predict_domain}-dev-norm-$((split - 1)).blogmel.scp \
    --param $model_dir/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    --print-logprob > $decode_dir/logprobs.$split.log

echo "Generating lattice..."
tail -n+2 $decode_dir/logprobs.$split.log \
    | python3 $UTILS/batch2ark.py \
    | latgen-faster-mapped \
    --max-active=7000 \
    --beam=15 \
    --lattice-beam=7 \
    --acoustic-scale=0.1 \
    --allow-partial=true \
    --word-symbol-table=$AMI/exp/ihm/tri3-logmel-hires/graph_ami_fsh.o3g.kn.pr1-7/words.txt \
    $AMI/exp/ihm/tri3-logmel-hires_ali/final.mdl \
    $AMI/exp/ihm/tri3-logmel-hires/graph_ami_fsh.o3g.kn.pr1-7/HCLG.fst \
    ark:- \
    "ark:|gzip -c > $decode_dir/lat.$split.gz" \
    2> $decode_log
echo "DONE AUGMENTED ACOUSTIC MODEL DECODING JOB"

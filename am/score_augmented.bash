#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1 #SBATCH -N1-1 #SBATCH -c 30 #SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J score_augmented

echo "STARTING AUGMENTED ACOUSTIC MODEL SCORING JOB"

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

expt_name="train_${tar_domain}/augmented_src_${src_domain}/${ARCH_NAME}/${CNN_NAME}"

if [ "$domain_adversarial" == true ]; then
    log_dir=$LOG_DIR/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    log_dir=$LOG_DIR/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    log_dir=$LOG_DIR/$expt_name/${run_mode}_ratio${NOISE_RATIO}
fi

decode_dir=$log_dir/decode_${predict_domain}
mkdir -p $decode_dir
score_log=$decode_dir/score.log

num_jobs=30
echo $num_jobs > $decode_dir/num_jobs

if [ "$DATASET_NAME" == "ami-0.1" ]; then
    data=$AMI/sls-data/ami-0.1/${predict_domain}-dev
elif [ "$DATASET_NAME" == "ami-full" ]; then
    data=$AMI/data/${predict_domain}/dev-logmel-hires
else
    echo "Unknown dataset $DATASET"
    exit 1
fi

lang=$AMI/exp/ihm/tri3-logmel-hires/graph_ami_fsh.o3g.kn.pr1-7
model=$AMI/exp/ihm/tri3-logmel-hires_ali/final.mdl

$MENG_ROOT/am/score_asclite.sh $data $lang $decode_dir $model > $score_log

echo "DONE AUGMENTED ACOUSTIC MODEL SCORING JOB"

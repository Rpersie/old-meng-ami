#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 30
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J score_combined

echo "STARTING COMBINED ACOUSTIC MODEL SCORING JOB"

. ./path.sh
. $MENG_ROOT/am/combined_config.sh
. $MENG_ROOT/am/path-opt.sh

if [ "$#" -lt 2 ]; then
    echo "Run mode and predict domain not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

predict_domain=$2
echo "Combined model (IHM baseline + IHM->SDM1 augmented), predict domain ${predict_domain}"

domain_adversarial=false
gan=false
if [ "$#" -ge 3 ]; then
    if [ "$3" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi
    
    if [ "$3" == "gan" ]; then
        gan=true
        echo "Using generative domain_adversarial net (GAN) style training"
    fi
fi

mkdir -p $LOG_DIR/$EXPT_NAME
if [ "$domain_adversarial" == true ]; then
    log_dir=$LOG_DIR/$EXPT_NAME/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    log_dir=$LOG_DIR/$EXPT_NAME/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    log_dir=$LOG_DIR/$EXPT_NAME/${run_mode}_ratio${NOISE_RATIO}
fi

decode_dir=$log_dir/decode_${predict_domain}
score_log=$decode_dir/score.log

num_jobs=30
echo $num_jobs > $decode_dir/num_jobs

if [ "$DATASET" == "ami-0.1" ]; then
    data=$AMI/sls-data/ami-0.1/${predict_domain}-dev
elif [ "$DATASET" == "ami-full" ]; then
    data=$AMI/data/${predict_domain}/dev-logmel-hires
else
    echo "Unknown dataset $DATASET"
    exit 1
fi

lang=$AMI/exp/ihm/tri3-logmel-hires/graph_ami_fsh.o3g.kn.pr1-7
model=$AMI/exp/ihm/tri3-logmel-hires_ali/final.mdl

$MENG_ROOT/am/score_asclite.sh $data $lang $decode_dir $model > $score_log

echo "DONE COMBINED ACOUSTIC MODEL SCORING JOB"

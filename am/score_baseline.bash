#!/bin/bash
#SBATCH -p cpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 30
#SBATCH --mem=32768
#SBATCH --time=24:00:00
#SBATCH -J score_baseline

echo "STARTING BASELINE ACOUSTIC MODEL SCORING JOB"

. ./path.sh
. $MENG_ROOT/am/baseline_config.sh
. $MENG_ROOT/am/path-opt.sh

if [ "$#" -lt 2 ]; then
    echo "Train domain and predict domain not specified; exiting"
    exit 1
fi

train_domain=$1
predict_domain=$2
echo "Train domain ${train_domain}, predict domain ${predict_domain}"

expt_name="train_${train_domain}/baseline/${ARCH_NAME}"

model_dir=$MODEL_DIR/$expt_name

decode_dir=$LOG_DIR/$expt_name/decode_${predict_domain}
mkdir -p $decode_dir
score_log=$decode_dir/score.log

num_jobs=30
echo $num_jobs > $decode_dir/num_jobs

if [ "$DATASET_NAME" == "ami-0.1" ]; then
    data=$AMI/sls-data/ami-0.1/${predict_domain}-dev
elif [ "$DATASET_NAME" == "ami-full" ]; then
    data=$AMI/data/${predict_domain}/dev-logmel-hires
else
    echo "Unknown dataset $DATASET_NAME"
    exit 1
fi

lang=$AMI/exp/ihm/tri3-logmel-hires/graph_ami_fsh.o3g.kn.pr1-7
model=$AMI/exp/ihm/tri3-logmel-hires_ali/final.mdl

$MENG_ROOT/am/score_asclite.sh $data $lang $decode_dir $model > $score_log

echo "DONE BASELINE ACOUSTIC MODEL SCORING JOB"

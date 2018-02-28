#!/bin/bash
#SBATCH -p 630
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --mem=32768
#SBATCH --time=12:00:00
#SBATCH --array=1-30
#SBATCH -J decode_baseline

echo "STARTING BASELINE ACOUSTIC MODEL DECODING JOB"

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

split=$SLURM_ARRAY_TASK_ID

expt_name="train_${train_domain}/baseline/${ARCH_NAME}"

model_dir=$MODEL_DIR/$expt_name

decode_dir=$LOG_DIR/$expt_name/decode_${predict_domain}
mkdir -p $decode_dir
decode_log=$decode_dir/decode_${split}.log

if [ "$DATASET_NAME" == "ami-full" ]; then
    frame_split_file=$DATASET/split30/${predict_domain}-dev-logmel-hires-$((split - 1)).blogmel.scp
else
    frame_split_file=$DATASET/split30/${predict_domain}-dev-norm-$((split - 1)).blogmel.scp
fi

echo "Predicting log probabilities for split ${split}..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
# Only use these environment variables if on 630 or 520 machines -- 510s don't work with them!
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $frame_split_file \
    --param $model_dir/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    --print-logprob \
    | tail -n+2 \
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
echo "DONE BASELINE ACOUSTIC MODEL DECODING JOB"

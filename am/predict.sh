#!/bin/bash

. ../path-opt.sh

epoch=$1

exp=sls-exp/ihm0.1/frame-tdnn-450x7-step0.05

OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp sls-data/ami-0.1/ihm-dev-norm.blogmel.scp \
    --param $exp/param/param-$epoch \
    --label sls-data/ihm-pdfids.txt \
    > $exp/log/dev-$epoch.log


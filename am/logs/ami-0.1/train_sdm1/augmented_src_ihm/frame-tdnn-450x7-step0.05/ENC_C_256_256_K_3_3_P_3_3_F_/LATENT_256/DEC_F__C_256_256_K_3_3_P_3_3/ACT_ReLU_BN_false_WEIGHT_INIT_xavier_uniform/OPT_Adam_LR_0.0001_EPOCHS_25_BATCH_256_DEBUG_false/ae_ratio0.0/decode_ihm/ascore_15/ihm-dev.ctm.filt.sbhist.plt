## GNUPLOT command file
set samples 1000
set key 30.000000,90.000000
set xrange [0:1]
set yrange [0:100]
set ylabel '% Hypothesis Correct'
set xlabel 'Confidence Scores'
set title  'Scaled Binned Confidence scores for /data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_ihm/ascore_15/ihm-dev.ctm.filt'
set nogrid
set size 0.78,1
set nolabel
plot '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_ihm/ascore_15/ihm-dev.ctm.filt.sbhist.dat'  title 'True' with boxes, x*100 title 'Expected'
set size 1.0, 1.0
set key

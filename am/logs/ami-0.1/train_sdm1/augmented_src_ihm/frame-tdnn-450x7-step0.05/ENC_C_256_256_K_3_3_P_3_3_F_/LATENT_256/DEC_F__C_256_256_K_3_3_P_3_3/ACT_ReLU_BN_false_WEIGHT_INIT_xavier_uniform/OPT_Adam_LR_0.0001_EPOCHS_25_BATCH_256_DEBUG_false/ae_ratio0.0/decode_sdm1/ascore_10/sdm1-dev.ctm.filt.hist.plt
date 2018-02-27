set samples 1000
set xrange [0.000000:1.000000]
set autoscale y
set size 0.78, 1.0
set nogrid
set ylabel 'Counts'
set xlabel 'Confidence Measure'
set title  'Confidence scores for /data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_sdm1/ascore_10/sdm1-dev.ctm.filt'
plot '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_sdm1/ascore_10/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%f' title 'All Conf.' with lines, \
     '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_sdm1/ascore_10/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%*s%f' title 'Correct Conf.' with lines, \
     '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/augmented_src_ihm/frame-tdnn-450x7-step0.05/ENC_C_256_256_K_3_3_P_3_3_F_/LATENT_256/DEC_F__C_256_256_K_3_3_P_3_3/ACT_ReLU_BN_false_WEIGHT_INIT_xavier_uniform/OPT_Adam_LR_0.0001_EPOCHS_25_BATCH_256_DEBUG_false/ae_ratio0.0/decode_sdm1/ascore_10/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%*s%*s%f' title 'Incorrect Conf.' with lines
set size 1.0, 1.0

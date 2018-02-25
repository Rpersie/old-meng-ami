set samples 1000
set xrange [0.000000:1.000000]
set autoscale y
set size 0.78, 1.0
set nogrid
set ylabel 'Counts'
set xlabel 'Confidence Measure'
set title  'Confidence scores for /data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_9/sdm1-dev.ctm.filt'
plot '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_9/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%f' title 'All Conf.' with lines, \
     '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_9/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%*s%f' title 'Correct Conf.' with lines, \
     '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_sdm1/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_9/sdm1-dev.ctm.filt.hist.dat' using 1:2 '%f%*s%*s%f' title 'Incorrect Conf.' with lines
set size 1.0, 1.0

## GNUPLOT command file
set samples 1000
set key 30.000000,90.000000
set xrange [0:1]
set yrange [0:100]
set ylabel '% Hypothesis Correct'
set xlabel 'Confidence Scores'
set title  'Scaled Binned Confidence scores for /data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_ihm/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_14/sdm1-dev.ctm.filt'
set nogrid
set size 0.78,1
set nolabel
plot '/data/sls/r/u/atitus5/meng/am/logs/ami-0.1/train_ihm/baseline/frame-tdnn-450x7-step0.05/decode_sdm1/ascore_14/sdm1-dev.ctm.filt.sbhist.dat'  title 'True' with boxes, x*100 title 'Expected'
set size 1.0, 1.0
set key

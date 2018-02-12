#!/usr/bin/env python3

import sys

import os
import numpy as np
sys.path.append("./")
from utils.hao_data import write_kaldi_hao_ark, write_kaldi_hao_scp

area = 'head'

with open(sys.argv[1], 'r') as pred:
    # Skip first line added to output by Hao script, so that we only read predictions
    pred.readline()

    with open(sys.argv[2], 'r') as gold:
        output_dir = sys.argv[3]

        print("Writing ARK...")
        with open(os.path.join(output_dir, "errors.ark"), 'w') as ark_fd:
            current_utt_id = None
            for ell1, ell2 in zip(pred, gold):
                if area == 'head':
                    current_utt_id = ell1.rstrip('\n')
                    area = 'body'

                elif area == 'body' and ell1 == '.\n':
                    current_utt_id = None
                    area = 'head'

                elif area == 'body':
                    pred_ids = ell1.split()
                    gold_ids = ell2.split()

                    # 0 indicates incorrect, 1 indicates correct
                    errors_array = np.zeros((1, len(pred_ids)))

                    row = 0
                    for p, q in zip(pred_ids, gold_ids):
                        if p == q:
                            errors_array[0, row] = 1
                        row += 1

                    write_kaldi_hao_ark(ark_fd, current_utt_id, errors_array)
        print("Done writing ARK")
        
        print("Writing SCP...")
        with open(os.path.join(output_dir, "errors.scp"), 'w') as scp_fd:
            write_kaldi_hao_scp(scp_fd, os.path.join(output_dir, "errors.ark"))
        print("Done writing SCP")

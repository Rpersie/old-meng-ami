#!/usr/bin/env python3

import sys

pred = open(sys.argv[1])
gold = open(sys.argv[2])

total_error = 0
nutt = 0

area = 'head'

# Skip first line added to output by Hao script, so that we only read predictions
pred.readline()

for ell1, ell2 in zip(pred, gold):
    if area == 'head':
        area = 'body'

    elif area == 'body' and ell1 == '.\n':
        area = 'head'

    elif area == 'body':
        pred_ids = ell1.split()
        gold_ids = ell2.split()

        error = 0

        for p, q in zip(pred_ids, gold_ids):
            if p != q:
                error += 1

        total_error += error / len(pred_ids)

        nutt += 1

print('utt: {} error rate: {}'.format(nutt, total_error / nutt))

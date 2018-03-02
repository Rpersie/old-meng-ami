#!/usr/bin/env python3

import re
import sys

k = int(sys.argv[1])

loss_sum = 0
samples = 0

last_k = []

for line in sys.stdin:

    parts = line.split()

    if parts and parts[0] == 'E:':
        if 'inf' in parts[1]:
            continue

        loss = float(parts[1])
        loss_sum += loss
        samples += 1

        last_k.append(loss)
        if len(last_k) > k:
            del last_k[0]

print('avg E over {} samples: {}'.format(samples, loss_sum / samples))
print('last {} E loss: {}'.format(len(last_k), sum(last_k) / len(last_k)))

#!/usr/bin/env python3

import random
import sys
import json
import math

sys.path.append("./am")
from nninit import gen_mat, gen_vec

layer = 7
# ninput = 140
ninput = 80
nhidden = 450
# npred = 3984    # IHM
npred = 3966    # SDM1

print('-1 0 1')
print('-1 0 1')
print('-1 0 1')
print('-3 0 3')
print('-3 0 3')
print('-3 0 3')
print('-3 0 3')
print('#')
    
zero = lambda d: 0.0

if sys.argv[1] == 'random':
    gen = lambda d: random.uniform(-math.sqrt(6 / d), math.sqrt(6 / d))
    ggen = lambda z: lambda d: random.uniform(-math.sqrt(6 / z), math.sqrt(6 / z))
else:
    gen = zero
    ggen = lambda z: zero

for i in range(layer):
    if i == 0:
        gen_mat(ninput, 3 * nhidden, gen)
        gen_vec(nhidden, zero)
    else:
        gen_mat(nhidden, 3 * nhidden, gen)
        gen_vec(nhidden, zero)

# softmax
gen_mat(nhidden, npred, ggen(nhidden))
gen_vec(npred, zero)


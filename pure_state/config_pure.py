#!/usr/bin/env python

"""
    config_pure.py: the configuration for pure state learning task

"""

import numpy as np
label = 'pure_state'

# system settings
system_size = [3]

#-----constances
lamb = np.float(2)

s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2

# learning scripts
initial_eta = 1e-1
epochs = 300
decay = False
eta = initial_eta
step_size = 1
replications = 1

# file settings
figure_path = './figure'
model_gen_path = './saved_model/{}qubit_model-gen(pure).mdl'.format(system_size)
model_dis_path = './saved_model/{}qubit_model-dis(pure).mdl'.format(system_size)


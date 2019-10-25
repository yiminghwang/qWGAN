#!/usr/bin/env python

"""
    config_mixed.py: the configuration for mixed state learning task

"""

import numpy as np
from tools.utils import get_zero_state

# Learning Scripts
# Regularization Parameters

lamb = np.float(10)
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = s ** 2 / 4 + s + 1
cst2 = s ** 2 / 4 + s / 2
cst3 = s ** 2 / 4

# Learning Scripts
initial_eta = 1e-1
epochs = 10
decay = False
eta = initial_eta

step_size = 1
prob_gen_lr = eta
theta_lr = eta
psi_lr = eta
phi_lr = eta


label = 'mixed_state'

fidelities = list()
losses = list()

# System setting//
system_size = 2

num_to_mix = 2

zero_state = get_zero_state(system_size)

# file settings
figure_path = './figure'
model_gen_path = './saved_model/{}qubit_model-gen(mixed).mdl'.format(system_size)
model_dis_path = './saved_model/{}qubit_model-dis(mixed).mdl'.format(system_size)


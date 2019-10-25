#!/usr/bin/env python

"""
    config_hs.py: the configuration for hamiltonian simulation task

"""

import numpy as np

# Learning Scripts
from tools.utils import get_maximally_entangled_state

lamb = np.float(10)
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2


# Learning Scripts
initial_eta = 1e-1
epochs = 300
decay = False
eta = initial_eta

# Log
label = 'hs'
fidelities = list()
losses = list()

# System setting//
system_size = 3
layer = 4

input_state = get_maximally_entangled_state(system_size)

# file settings
figure_path = './figure'
model_gen_path = './saved_model/{}qubit_model-gen(hs).mdl'.format(system_size)
model_dis_path = './saved_model/{}qubit_model-dis(hs).mdl'.format(system_size)


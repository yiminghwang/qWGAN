#!/usr/bin/env python

"""
    config_noise.py: the configuration for noise model

"""

import numpy as np

# constants
lamb = np.float(10)
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = s ** 2 / 4 + s + 1
cst2 = s ** 2 / 4 + s / 2
cst3 = s ** 2 / 4

# Learning Scripts
initial_eta = 1e-1
epochs = 500
decay = False
eta = initial_eta

step_size = 1
prob_gen_lr = eta
theta_lr = eta
psi_lr = eta
phi_lr = eta

# Log
label = 'noise'

# System setting//
system_size = 4
num_to_mix = 1

# Noise setting
sigma = 0.05
mu = 0

# file settings
figure_path = './figure'
model_gen_path = './saved_model/{}qubit_model-gen(noise).mdl'.format(system_size)
model_dis_path = './saved_model/{}qubit_model-dis(noise).mdl'.format(system_size)
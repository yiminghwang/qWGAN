#!/usr/bin/env python

"""
    training_hs.py: hamiltonian simulation by qwgan framework

"""

import time
from datetime import datetime
from model.model_hs import Generator, Discriminator, compute_cost, compute_fidelity
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import *
import config_hs as cf
from tools.utils import save_model, train_log, get_maximally_entangled_state
import scipy.io as scio

np.random.seed()

def construct_qcircuit(qc,size,layer):
    entg_list = ["XX", "YY", "ZZ"]
    for j in range(layer):
        for i in range(size):
            if i < size - 1:
                for gate in entg_list:
                    qc.add_gate(Quantum_Gate(gate, i, i + 1, angle=0.5000 * np.pi))
                qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        for gate in entg_list:
            qc.add_gate(Quantum_Gate(gate, 0, size - 1, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate("G", None, angle=0.5000 * np.pi))

    theta = np.random.randn(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc


def main():

    input_state = get_maximally_entangled_state(cf.system_size)
    
    target_unitary = scio.loadmat('./exp_ideal_{}_qubit.mat'.format(cf.system_size))['exp_ideal']
    real_state_tmp = np.matmul(np.kron(target_unitary, Identity(cf.system_size)), input_state)
    real_state = real_state_tmp

    step_size = 1
    
    # define generator
    gen = Generator(cf.system_size)
    gen.set_qcircuit(construct_qcircuit(gen.qc, cf.system_size, cf.layer))

    # define discriminator
    herm = [I, X, Y, Z]
    dis = Discriminator(herm, cf.system_size*2)

    f = compute_fidelity(gen, input_state, real_state)

    fidelities = []
    losses = []

    while (f < 0.99):
        fidelities[:] = []
        losses[:] = []
        starttime = datetime.now()
        for iter in range(cf.epochs):
            print("==================================================")
            print("Epoch {}, Step_size {}".format(iter + 1, cf.eta))

            if iter % step_size == 0:
                # Generator gradient descent
                gen.update_gen(dis, real_state)
                print("Loss after generator step: {}".format(compute_cost(gen, dis, real_state)))

            # Discriminator gradient ascent
            dis.update_dis(gen, real_state)
            print("Loss after discriminator step: {}".format(compute_cost(gen, dis, real_state)))

            cost = compute_cost(gen, dis, real_state)
            fidelity = compute_fidelity(gen, input_state, real_state)
            losses.append(cost)
            fidelities.append(fidelity)

            if iter % 10 ==0:
                endtime = datetime.now()
                training_duration = (endtime - starttime).seconds/np.float(3600)
                param = 'epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n'.format(iter,round(fidelity,6),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),round(training_duration,2))
                train_log(param, './{}qubit_log_hs.txt'.format(cf.system_size))

            print("Fidelity between real and fake state: {}".format(fidelity))
            print("==================================================")

            if (cf.decay):
                cf.eta = (cf.initial_eta * (cf.epochs - iter - 1) +
                       (1e-2 * cf.initial_eta) * iter) / cf.epochs
        f = compute_fidelity(gen, input_state, real_state)

    plt_fidelity_vs_iter(fidelities, losses, cf)
    save_model(gen, cf.model_gen_path)
    save_model(dis, cf.model_dis_path)

    print("end")

if __name__ == '__main__':
    main()
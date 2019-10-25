#!/usr/bin/env python

"""
    training_mixed_state.py: training process of qwgan for mixed state

"""
import time
from datetime import datetime
from model.model_mixed import Generator, Discriminator, compute_cost, compute_fidelity
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import *
import config_mixed as cf
from tools.utils import get_zero_state, save_model, getreal_denmat, train_log

np.random.seed()

def construct_qcircuit(qc_list,size):
    '''
        the function to construct quantum circuit of generator
    :param qc:
    :param size:
    :return:
    '''
    for qc in qc_list:
        for i in range(size):
            qc.add_gate(Quantum_Gate("X", i, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("Y", i, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))

        theta = np.random.random(len(qc.gates))
        for i in range(len(qc.gates)):
            qc.gates[i].angle = theta[i]
    return qc_list

def main():
    losses = []
    fidelities = []

    zero_state = get_zero_state(cf.system_size)

    # Real_state
    input_state = list()
    angle = np.random.randint(1,10,size=[cf.num_to_mix,cf.system_size,3])
    for i in range(cf.num_to_mix):
        matrix = Identity(cf.system_size)
        for j in range(cf.system_size):
            row_i_mat = np.matmul(Z_Rotation(cf.system_size, j, np.pi * angle[i][j][2], False),
                                  np.matmul(Y_Rotation(cf.system_size, j, np.pi * angle[i][j][1], False),
                                            X_Rotation(cf.system_size, j, np.pi * angle[i][j][0], False)))
            matrix = np.matmul(row_i_mat, matrix)
        state = np.matmul(matrix, zero_state)
        input_state.append(np.asmatrix(state))
    prob_real = [0.2, 0.8]
    real_state = getreal_denmat(cf,prob_real,input_state)

    # define generator
    gen = Generator(cf.system_size, cf.num_to_mix)
    gen.set_qcircuit(construct_qcircuit(gen.qc_list, cf.system_size))

    # define discriminator
    herm = [I, X, Y, Z]
    dis = Discriminator(herm, cf.system_size)
    f = compute_fidelity(gen, zero_state, real_state)

    while (f < 0.99):
        starttime = datetime.now()
        for iter in range(cf.epochs):
            print("==================================================")
            print("Epoch {}, Step_size {}".format(iter + 1, cf.eta))

            if iter % cf.step_size == 0:
                # Generator gradient descent
                gen.update_gen(dis,real_state)
                print("Loss after generator step: {}".format(compute_cost(gen, dis,real_state)))

            # Discriminator gradient ascent
            dis.update_dis(gen,real_state)
            print("Loss after discriminator step: {}".format(compute_cost(gen, dis,real_state)))

            cost = compute_cost(gen, dis,real_state)
            fidelity = compute_fidelity(gen, zero_state,real_state)

            losses.append(cost)
            fidelities.append(fidelity)

            print("Fidelity between real and fake state: {}".format(fidelity))
            print("==================================================")

            if iter % 10 == 0:
                endtime = datetime.now()
                training_duration = (endtime - starttime).seconds / np.float(3600)
                param = 'epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n'.format(iter, round(fidelity, 6),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),round(training_duration, 2))
                train_log(param, './{}qubit_log_mixed.txt'.format(cf.system_size))

            if (cf.decay):
                cf.eta = (cf.initial_eta * (cf.epochs - iter - 1) +
                       (1e-2 * cf.initial_eta) * iter) / cf.epochs

        f = compute_fidelity(gen, zero_state, real_state)

    plt_fidelity_vs_iter(fidelities, losses, cf)
    save_model(gen, cf.model_gen_path)
    save_model(dis, cf.model_dis_path)

    fidelities[:] = []
    losses[:] = []
    print("end")


if __name__ == '__main__':
    main()
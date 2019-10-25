#!/usr/bin/env python

"""
    noise_qwgan.py: 4 qubits pure state noise model

"""
import time
from datetime import datetime
from model.model_noise import Gen, Dis, compute_cost, compute_fidelity
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import *
import config_noise as cf
from tools.utils import get_zero_state, save_model, train_log

np.random.seed()

def construct_qcircuit(qc_list):
    '''
        the function to construct quantum circuit of generator
    :param qc:
    :return:
    '''
    for qc in qc_list:
        qc.add_gate(Quantum_Gate("X", 0, angle=0.12300 * np.pi))
        qc.add_gate(Quantum_Gate("X", 1, angle=-0.23000 * np.pi))
        qc.add_gate(Quantum_Gate("X", 2, angle=-0.5400 * np.pi))
        qc.add_gate(Quantum_Gate("X", 3, angle=0.45000 * np.pi))

        qc.add_gate(Quantum_Gate("Y", 0, angle=0.5050 * np.pi))
        qc.add_gate(Quantum_Gate("Y", 1, angle=0.1000 * np.pi))
        qc.add_gate(Quantum_Gate("Y", 2, angle=-0.5400 * np.pi))
        qc.add_gate(Quantum_Gate("Y", 3, angle=0.7230 * np.pi))

        qc.add_gate(Quantum_Gate("Z", 0, angle=0.2000 * np.pi))
        qc.add_gate(Quantum_Gate("Z", 1, angle=0.4200 * np.pi))
        qc.add_gate(Quantum_Gate("Z", 2, angle=-0.8600 * np.pi))
        qc.add_gate(Quantum_Gate("Z", 3, angle=0.0200 * np.pi))

        qc.add_gate(Quantum_Gate("XX", 0, 1, angle=-0.21500 * np.pi))
        qc.add_gate(Quantum_Gate("XX", 0, 2, angle=0.6430 * np.pi))
        qc.add_gate(Quantum_Gate("XX", 0, 3, angle=-0.83500 * np.pi))
        qc.add_gate(Quantum_Gate("XX", 1, 2, angle=-0.104700 * np.pi))
        qc.add_gate(Quantum_Gate("XX", 1, 3, angle=0.757200 * np.pi))
        qc.add_gate(Quantum_Gate("XX", 2, 3, angle=-0.58200 * np.pi))

        theta = np.random.rand(len(qc.gates))
        for i in range(len(qc.gates)):
            qc.gates[i].angle = theta[i]

    return qc_list


def main():
    angle = np.random.randint(1,10,size=[cf.system_size,3])
    matrix = Identity(cf.system_size)
    for j in range(cf.system_size):
        row_i_mat = np.matmul(Z_Rotation(cf.system_size, j, np.pi * angle[j][2], False),
                                  np.matmul(Y_Rotation(cf.system_size, j, np.pi * angle[j][1], False),
                                            X_Rotation(cf.system_size, j, np.pi * angle[j][0], False)))
        matrix = np.matmul(row_i_mat, matrix)

    param = np.random.rand(6)
    XX1 = XX_Rotation(cf.system_size, 0, 1, param[0], False)
    XX2 = XX_Rotation(cf.system_size, 0, 2, param[1], False)
    XX3 = XX_Rotation(cf.system_size, 0, 3, param[2], False)
    XX4 = XX_Rotation(cf.system_size, 1, 2, param[3], False)
    XX5 = XX_Rotation(cf.system_size, 1, 3, param[4], False)
    XX6 = XX_Rotation(cf.system_size, 2, 3, param[5], False)

    zero_state = get_zero_state(cf.system_size)

    real_state_tmp = np.matmul(XX6 ,np.matmul( XX5 ,np.matmul( XX4 ,np.matmul( XX3 ,np.matmul(XX2 , np.matmul(XX1 ,np.matmul( matrix , zero_state)))))))
    real_state = np.matmul(real_state_tmp , real_state_tmp.getH())

    starttime = datetime.now()

    # define generator
    gen = Gen(cf.system_size, cf.num_to_mix, cf.mu, cf.sigma)
    gen.set_qcircuit(construct_qcircuit(gen.qc_list))

    # define discriminator
    herm = [I, X, Y, Z]
    dis = Dis(herm, cf.system_size, cf.mu,cf.sigma)

    fidelities = list()
    losses = list()

    f = compute_fidelity(gen, zero_state, real_state)

    while (f < 0.99):

        starttime = datetime.now()
        for iter in range(cf.epochs):
            print("==================================================")
            print("Epoch {}, Step_size {}".format(iter + 1, cf.eta))

            compute_cost(gen, dis, real_state)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            if iter % cf.step_size == 0:
                # Generator gradient descent
                gen.update_gen(dis,real_state)
                print("Loss after generator step: {}".format(compute_cost(gen, dis,real_state)))

            # Discriminator gradient ascent
            dis.update_dis(gen,real_state)
            print("Loss after discriminator step: {}".format(compute_cost(gen, dis, real_state)))

            cost = compute_cost(gen, dis, real_state)
            fidelity = compute_fidelity(gen, zero_state,real_state)

            losses.append(cost)
            fidelities.append(fidelity)

            print("Fidelity between real and fake state: {}".format(fidelity))
            print("==================================================")

            if iter % 10 == 0:
                endtime = datetime.now()
                training_duration = (endtime - starttime).seconds / np.float(3600)
                param = 'epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n'.format(iter, round(fidelity, 6),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),round(training_duration, 2))
                train_log(param, './{}qubit_log_noise.txt'.format(cf.system_size))

            if (cf.decay):
                eta = (cf.initial_eta * (cf.epochs - iter - 1) +
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
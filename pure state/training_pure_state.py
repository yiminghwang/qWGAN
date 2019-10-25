#!/usr/bin/env python

"""
    training_pure_state.py: training process of qwgan for pure state

"""
import time
from datetime import datetime
from model.model_pure import Generator, Discriminator, compute_fidelity, compute_cost, get_zero_state
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import *
import config_pure as cf
from tools.utils import save_model, train_log

np.random.seed()

def construct_qcircuit(qc,size):
    '''
        the function to construct quantum circuit of generator
    :param qc:
    :param size:
    :return:
    '''
    for i in range(size):
        qc.add_gate(Quantum_Gate("X", i, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate("Y", i, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))

    theta = np.random.random(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc

def main():

    for size in cf.system_size:

        zero_state = get_zero_state(size)

        fidelities = list()
        losses = list()

        for i in range(cf.replications):

            angle = np.random.randint(1, 10, size=[size, 3])
            matrix = Identity(size)
            for j in range(size):
                row_i_mat = np.matmul(Z_Rotation(size, j, np.pi / angle[j][2], False),
                                      np.matmul(Y_Rotation(size, j, np.pi / angle[j][1], False),
                                                X_Rotation(size, j, np.pi / angle[j][0], False)))
                matrix = np.matmul(row_i_mat, matrix)
            real_state = np.matmul(matrix, zero_state)

            # define generator
            gen = Generator(size)
            gen.set_qcircuit(construct_qcircuit(gen.qc,size))

            # define discriminator
            herm = [I, X, Y, Z]

            dis = Discriminator(herm, size)

            f = compute_fidelity(gen,zero_state,real_state)
            # optional term, this is for controlling the initial fidelity is small.
            # while(compute_fidelity(gen,zero_state,real_state)>0.5):
            #     gen.reset_angles()
            while(compute_fidelity(gen,zero_state,real_state)<0.001):
                gen.reset_angles()

            while(f < 0.99):
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

                    cost = compute_cost(gen, dis, real_state)
                    fidelity = compute_fidelity(gen, zero_state, real_state)

                    losses.append(cost)
                    fidelities.append(fidelity)

                    print("Fidelity between real and fake state: {}".format(fidelity))
                    print("==================================================")

                    if iter % 10 == 0:
                        endtime = datetime.now()
                        training_duration = (endtime - starttime).seconds / np.float(3600)
                        param = 'epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n'.format(iter,round(fidelity,6),time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),round(training_duration,2))
                        train_log(param, './{}qubit_log_pure.txt'.format(cf.system_size))

                    if (cf.decay):
                        eta = (cf.initial_eta * (cf.epochs - iter - 1) +
                               (cf.initial_eta) * iter) / cf.epochs

                f = compute_fidelity(gen,zero_state,real_state)

            plt_fidelity_vs_iter(fidelities, losses, cf, indx=i)
            save_model(gen, cf.model_gen_path)
            save_model(dis, cf.model_dis_path)
            
            fidelities[:]=[]
            losses[:]=[]
    print("end")

if __name__ == '__main__':

    main()



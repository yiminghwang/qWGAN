#!/usr/bin/env python

"""
    exprmt-entgl-paraherm.py: including more flexible methods for qgan

"""


__author__ = "yiming huang"
__email__ = "yiminghwang@gmail.com"
__version__ = '0.0.1'
__date__ = '27/03/2019'
__status__ = "BETA"

from base_notation import *
from scipy.sparse import dok_matrix
import matplotlib
import numpy as np
import time
from datetime import datetime
from momentum import MomentumOptimizer
import pickle
from scipy.linalg import expm, sqrtm, eigh
import scipy.io as scio
import pandas as pd
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


np.random.seed()


def getCmatrix(system_size):
    '''
            Cmatrix = ( I -  SWAP )/2
    :param system_sizes: # total qubits of psi and phi
    :return: Cmatrix(matrix)
    '''
    dim = 2 ** system_size
    SWAP = dok_matrix((dim * dim, dim * dim))

    for i in range(dim):
        for j in range(dim):
            SWAP[i * dim + j, j * dim + i] = 1
            SWAP[j * dim + i, i * dim + j] = 1
    return np.asmatrix((np.eye(dim * dim) - SWAP) / 2)
    # return SWAP


def getfake_dens_mat(G, state):
    f_state = np.matmul(G, state)
    f_denmat = np.matmul(f_state, f_state.getH())
    return f_denmat


def compute_cost(gen, dis):
    G = gen.getGen()
    psi = dis.getPsi()
    phi = dis.getPhi()

    fake_state = np.matmul(G, zero_state)

    try:
        A = expm(np.float(-1 / lamb) * phi)
    except Exception:
        print('cost function -1/lamb:\n', (-1 / lamb))
        print('size of phi:\n', phi.shape)

    try:
        B = expm(np.float(1 / lamb) * psi)
    except Exception:
        print('cost function 1/lamb:\n', (1 / lamb))
        print('size of psi:\n', psi.shape)

    term1 = np.matmul(fake_state.getH(), np.matmul(A, fake_state))
    term2 = np.matmul(real_state.getH(), np.matmul(B, real_state))

    term3 = np.matmul(fake_state.getH(), np.matmul(B, real_state))
    term4 = np.matmul(real_state.getH(), np.matmul(A, fake_state))

    term5 = np.matmul(fake_state.getH(), np.matmul(A, real_state))
    term6 = np.matmul(real_state.getH(), np.matmul(B, fake_state))

    term7 = np.matmul(fake_state.getH(), np.matmul(B, fake_state))
    term8 = np.matmul(real_state.getH(), np.matmul(A, real_state))

    # psiterm = np.asscalar(real_state.getH() @ psi @ real_state)
    # phiterm = np.asscalar(fake_state.getH() @ phi @ fake_state)
    psiterm = np.trace(np.matmul(np.matmul(real_state, real_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(fake_state, fake_state.getH()), phi))

    regterm = np.asscalar(
        lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8))

    # -------------original regterm
    # join_state = np.kron(fake_state, real_state)
    #
    # Eterm = expm(np.asarray((-Cmatrix + np.kron(np.eye(2 ** system_size), psi) - np.kron(phi, np.eye(2 ** system_size))) / lamb))
    #
    # regterm_orignal = lamb/np.e * (join_state.getH()@Eterm@join_state)
    #
    # # print(regterm - regterm_orignal)
    # print('psiterm:', psiterm)
    # print('phiterm:', phiterm)
    # print('regterm:', regterm)
    loss = np.real(psiterm - phiterm - regterm)

    # del G,phi,psi,A,B,psiterm,phiterm,regterm
    # gc.collect()

    return loss

    # return psiterm - phiterm - regterm


def compute_fidelity(gen, input_state, real_state, type='training'):
    '''
        calculate the fidelity between target state and fake state
    :param gen: generator(Generator)
    :param state: vector(array), input state
    :return:
    '''

    ## for density matrix
    # G = gen.getGen()
    # fake_state = getfake_dens_mat(G, state)
    # tmp = sqrtm(fake_state)
    # fidelity = sqrtm(tmp @ real_state @ tmp)
    # return np.real(np.square(np.trace(fidelity)))
    if type == 'test':
        G = gen.qc.get_mat_rep()
    else:
        G = gen.getGen()
    fake_state = np.matmul(G, input_state)
    return np.abs(np.asscalar(np.matmul(real_state.getH(), fake_state))) ** 2

class Generator:
    def __init__(self, system_size):
        self.size = system_size
        self.qc = self.init_qcircuit()
        self.optimizer = MomentumOptimizer()

    def single_layer(self,qc):

        for i in range(self.size):
            # qc.add_gate(Quantum_Gate("X", i, angle=0.5000 * np.pi))
            # qc.add_gate(Quantum_Gate("Y", i, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        return qc

    def entanglement_layer(self,qc):

        for i in range(self.size):
            for j in range(i + 1, self.size):
                # print(i,j)
                qc.add_gate(Quantum_Gate("XX", i, j, angle=0.5000 * np.pi))
                qc.add_gate(Quantum_Gate("YY", i, j, angle=0.5000 * np.pi))
                qc.add_gate(Quantum_Gate("ZZ", i, j, angle=0.5000 * np.pi))
                print(i,j)
        return qc

    def entangle_adjacent_layer(self,qc,type):

        for i in range(self.size-1):
            qc.add_gate(Quantum_Gate(type, i, i+1, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate(type, 0, self.size - 1, angle=0.5000 * np.pi))
        return qc

    def set_qcircuit(self, qc):

        # qc_tmp = qc
        # for i in range(4):
        #     qc_tmp = self.single_layer(qc_tmp)
        #     qc_tmp = self.entanglement_layer(qc_tmp)
        # qc = qc_tmp

        # for j in range(layer):
        #     for i in range(self.size):
        #         qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        #         if i < self.size:
        #             qc = self.entangle_adjacent_layer(qc, "XX")
        #             qc = self.entangle_adjacent_layer(qc, "YY")
        #             qc = self.entangle_adjacent_layer(qc, "ZZ")
        #     qc.add_gate(Quantum_Gate("G",None,angle=0.5000 * np.pi))
        entg_list = ["XX","YY","ZZ"]
        for j in range(layer):
            for i in range(self.size):
                if i < self.size-1:
                    for gate in entg_list:
                        qc.add_gate(Quantum_Gate(gate, i, i+1, angle=0.5000 * np.pi))
                    qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
            for gate in entg_list:
               qc.add_gate(Quantum_Gate(gate, 0, self.size-1, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("G", None, angle=0.5000 * np.pi))

        # for gate in entg_list:
        #     for i in range(self.size):
        #         if i < self.size - 1:
        #             qc.add_gate(Quantum_Gate(gate, i, i + 1, angle=0.5000 * np.pi))
        #         else:
        #             qc.add_gate(Quantum_Gate(gate, 0, self.size - 1, angle=0.5000 * np.pi))
        #         qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))

        theta = np.random.randn(len(qc.gates))
        for i in range(len(qc.gates)):
            qc.gates[i].angle = theta[i]
            print(i+1,qc.gates[i].name,qc.gates[i].qubit1,qc.gates[i].qubit2)

        return qc

    def init_qcircuit(self):
        qcircuit = Quantum_Circuit(self.size, "generator")
        self.set_qcircuit(qcircuit)
        return qcircuit

    def getGen(self):
        return np.kron(self.qc.get_mat_rep(), Identity(system_size))

    def _grad_gen(self, dis):

        G = self.getGen()

        phi = dis.getPhi()
        psi = dis.getPsi()

        fake_state = np.matmul(G, zero_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print('grad_gen -1/lamb:\n', (-1 / lamb))
            print('size of phi:\n', phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print('grad_gen 1/lamb:\n', (1 / lamb))
            print('size of psi:\n', psi.shape)

        # print("g: \n", G)
        # print("phi: \n", phi)
        # print("psi: \n", psi)

        # phi_tmp = np.asmatrix(phi)
        # print('phi:\n',phi_tmp.getH()@phi_tmp)
        #
        # psi_tmp = np.asmatrix(psi)
        # print('psi:\n',psi_tmp.getH()@psi_tmp)

        # print("expHerm:",expHerm)

        grad_g_psi = list()
        grad_g_phi = list()
        grad_g_reg = list()

        for i in range(self.qc.depth):

            grad_i = np.kron(self.qc.get_grad_mat_rep(i), Identity(system_size))
            # for psi term
            grad_g_psi.append(0)

            # for phi term
            fake_grad = np.matmul(grad_i, zero_state)
            tmp_grad = np.matmul(fake_grad.getH(), np.matmul(phi, fake_state)) + np.matmul(fake_state.getH(),np.matmul(phi, fake_grad))

            grad_g_phi.append(np.asscalar(tmp_grad))

            # for reg term

            term1 = np.matmul(fake_grad.getH(), np.matmul(A, fake_state)) * np.matmul(real_state.getH(),np.matmul(B, real_state))
            term2 = np.matmul(fake_state.getH(), np.matmul(A, fake_grad)) * np.matmul(real_state.getH(),np.matmul(B, real_state))

            term3 = np.matmul(fake_grad.getH(), np.matmul(B, real_state)) * np.matmul(real_state.getH(),np.matmul(A, fake_state))
            term4 = np.matmul(fake_state.getH(), np.matmul(B, real_state)) * np.matmul(real_state.getH(),np.matmul(A, fake_grad))

            term5 = np.matmul(fake_grad.getH(), np.matmul(A, real_state)) * np.matmul(real_state.getH(),np.matmul(B, fake_state))
            term6 = np.matmul(fake_state.getH(), np.matmul(A, real_state)) * np.matmul(real_state.getH(),np.matmul(B, fake_grad))

            term7 = np.matmul(fake_grad.getH(), np.matmul(B, fake_state)) * np.matmul(real_state.getH(),np.matmul(A, real_state))
            term8 = np.matmul(fake_state.getH(), np.matmul(B, fake_grad)) * np.matmul(real_state.getH(),np.matmul(A, real_state))

            tmp_reg_grad = lamb / np.e * (
                    cst1 * (term1 + term2) - cst2 * (term3 + term4) - cst2 * (term5 + term6) + cst3 * (term7 + term8))

            grad_g_reg.append(np.asscalar(tmp_reg_grad))

        g_psi = np.asarray(grad_g_psi)
        g_phi = np.asarray(grad_g_phi)
        g_reg = np.asarray(grad_g_reg)

        grad = np.real(g_psi - g_phi - g_reg)

        # print("grad:\n",grad)
        # del grad_g,grad_g_phi,grad_g_psi,grad_g_reg,G,phi,psi,A,B,fake_grad,tmp_grad,g_phi,g_psi,g_reg,grad_i
        # gc.collect()

        return grad

    def update_gen(self, dis):

        theta = []
        for gate in self.qc.gates:
            theta.append(gate.angle)

        grad = np.asarray(self._grad_gen(dis))
        theta = np.asarray(theta)
        new_angle = self.optimizer.compute_grad(theta,grad,'min')
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]

        print('gen_theta max:{}  gen_theta min:{}'.format(np.max(grad), np.min(grad)))

        # del grad_list,new_angle
        # gc.collect()


class Discriminator:

    def __init__(self, herm, system_size):
        self.size = system_size
        self.herm = herm
        self.alpha = np.zeros((self.size, len(self.herm)))
        self.beta = np.zeros((self.size, len(self.herm)))
        self._init_params()
        self.optimizer_psi = MomentumOptimizer()
        self.optimizer_phi = MomentumOptimizer()

    def _init_params(self):
        # Discriminator Parameters

        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

        # for i in range(self.size):
        #     self.alpha[i] = np.ones(len(self.herm))/16
        #     self.beta[i] = np.ones(len(self.herm))/16

    def getPsi(self):
        '''
            get matrix representation of real part of discriminator
        :param alpha:
                    parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        :return:
        '''
        psi = 1
        for i in range(self.size):
            psi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
            psi = np.kron(psi, psi_i)
        return psi

    def getPhi(self):
        '''
            get matrix representation of fake part of discriminator
        :param beta:
                    parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        :return:
        '''
        phi = 1
        for i in range(self.size):
            phi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                phi_i += self.beta[i][j] * self.herm[j]
            phi = np.kron(phi, phi_i)
        return phi

    # Psi gradients
    def _grad_psi(self, type):
        grad_psi = list()
        for i in range(self.size):
            grad_psiI = 1
            for j in range(self.size):
                if i == j:
                    grad_psii = self.herm[type]
                else:
                    grad_psii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_psii += self.alpha[j][k] * self.herm[k]
                grad_psiI = np.kron(grad_psiI, grad_psii)
            grad_psi.append(grad_psiI)
        return grad_psi

    def _grad_alpha(self, gen):
        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, zero_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print('grad_alpha -1/lamb:\n', (-1 / lamb))
            print('size of phi:\n', phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print('grad_alpha 1/lamb:\n', (1 / lamb))
            print('size of psi:\n', psi.shape)

        cs = 1 / lamb
        # print("g: \n", G)
        # print("phi: \n", phi)
        # print("psi: \n", psi)
        # print("expHerm:", expHerm)
        # print("fake_state:\n", fake_state)

        grad_psi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_phi_term = np.zeros_like(self.alpha, dtype=complex)
        grad_reg_term = np.zeros_like(self.alpha, dtype=complex)

        for type in range(len(self.herm)):
            gradpsi = self._grad_psi(type)

            gradpsi_list = list()
            gradphi_list = list()
            gradreg_list = list()

            for grad_psi in gradpsi:
                gradpsi_list.append(np.asscalar(np.matmul(real_state.getH(), np.matmul(grad_psi, real_state))))

                gradphi_list.append(0)

                term1 = cs * np.matmul(fake_state.getH(), np.matmul(A, fake_state)) * np.matmul(real_state.getH(),np.matmul(grad_psi,np.matmul(B,real_state)))
                term2 = cs * np.matmul(fake_state.getH(), np.matmul(grad_psi, np.matmul(B, real_state))) * np.matmul(real_state.getH(), np.matmul(A, fake_state))
                term3 = cs * np.matmul(fake_state.getH(), np.matmul(A, real_state)) * np.matmul(real_state.getH(),np.matmul(grad_psi,np.matmul(B,fake_state)))
                term4 = cs * np.matmul(fake_state.getH(), np.matmul(grad_psi, np.matmul(B, fake_state))) * np.matmul(real_state.getH(), np.matmul(A, real_state))

                gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)


        # del gradphi_list,gradpsi_list,gradreg_list,grad_psi_term,grad_phi_term,grad_reg_term,G,phi,psi,A,B
        # gc.collect()

        # print("grad_alpha:\n",np.real(grad_psi_term - grad_phi_term - grad_reg_term))
        return grad

    # Phi gradients
    def _grad_phi(self, type):
        grad_phi = list()
        for i in range(self.size):
            grad_phiI = 1
            for j in range(self.size):
                if i == j:
                    grad_phii = self.herm[type]
                else:
                    grad_phii = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        grad_phii += self.beta[j][k] * self.herm[k]
                grad_phiI = np.kron(grad_phiI, grad_phii)
            grad_phi.append(grad_phiI)
        return grad_phi

    def _grad_beta(self, gen):

        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, zero_state)

        try:
            A = expm((-1 / lamb) * phi)
        except Exception:
            print('grad_beta -1/lamb:\n', (-1 / lamb))
            print('size of phi:\n', phi.shape)

        try:
            B = expm((1 / lamb) * psi)
        except Exception:
            print('grad_beta 1/lamb:\n', (1 / lamb))
            print('size of psi:\n', psi.shape)

        cs = -1 / lamb

        grad_psi_term = np.zeros_like(self.beta, dtype=complex)
        grad_phi_term = np.zeros_like(self.beta, dtype=complex)
        grad_reg_term = np.zeros_like(self.beta, dtype=complex)

        for type in range(len(self.herm)):
            gradphi = self._grad_phi(type)

            gradpsi_list = list()
            gradphi_list = list()
            gradreg_list = list()

            for grad_phi in gradphi:
                gradpsi_list.append(0)

                gradphi_list.append(np.asscalar(np.matmul(fake_state.getH(), np.matmul(grad_phi, fake_state))))

                term1 = cs * np.matmul(fake_state.getH(), np.matmul(grad_phi, np.matmul(A, fake_state))) * np.matmul(real_state.getH(), np.matmul(B, real_state))
                term2 = cs * np.matmul(fake_state.getH(), np.matmul(B, real_state)) * np.matmul(real_state.getH(),np.matmul(grad_phi,np.matmul(A,fake_state)))
                term3 = cs * np.matmul(fake_state.getH(), np.matmul(grad_phi, np.matmul(A, real_state))) * np.matmul(real_state.getH(), np.matmul(B, fake_state))
                term4 = cs * np.matmul(fake_state.getH(), np.matmul(B, fake_state)) * np.matmul(real_state.getH(),np.matmul(grad_phi,np.matmul(A,real_state)))

                gradreg_list.append(np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)

        # del gradphi_list, gradpsi_list, gradreg_list,grad_psi_term,grad_phi_term,grad_reg_term,G,phi,psi,A,B
        # gc.collect()

        # print("grad_beta:\n",np.real(grad_psi_term - grad_phi_term - grad_reg_term))
        return grad

    def update_dis(self, gen):

        grad_alpha = self._grad_alpha(gen)
        # update alpha
        new_alpha = self.optimizer_psi.compute_grad(self.alpha, grad_alpha, 'max')
        # new_alpha = self.alpha + eta * self._grad_alpha(gen)

        grad_beta = self._grad_beta(gen)
        # update beta
        new_beta = self.optimizer_phi.compute_grad(self.beta, grad_beta, 'max')
        # new_beta = self.beta + eta * self._grad_beta(gen)

        self.alpha = new_alpha
        self.beta = new_beta

        print('alpha max:{} alpha min:{}'.format(np.max(self.alpha), np.min(self.alpha)))
        print('beta max:{} beta min:{}'.format(np.max(self.beta), np.min(self.beta)))

        print('grad_psi max:{}  grad_psi min:{}'.format(np.max(grad_alpha), np.min(grad_alpha)))
        print('grad_phi max:{}  grad_phi min:{}'.format(np.max(grad_beta), np.min(grad_beta)))


def train_log(param,file_path):
    with open(file_path, 'a') as file:
        file.write(param)

def save_model(gen, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(gen, file)

def load_model(file_path):
    with open(file_path, 'rb') as qc:
        model = pickle.load(qc)
    return model






# Learning Scripts
# Regularization Parameter
num = 1
lamb = np.float(10 ^ (num))
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2

# Learning Scripts
initial_eta = 1e-1
epochs = 1
decay = False
eta = initial_eta

# Log
label = 'hsc'
fidelities = list()
losses = list()

# System setting//
system_size = 5
layer = 1
Cmatrix = getCmatrix(system_size)

# zero_state = np.sqrt(np.ones(2 ** system_size)/(2 ** system_size))
# zero_state = np.zeros(2 ** system_size)
# zero_state[0] = 1
# zero_state = np.asmatrix(zero_state).T
# print(zero_state)

zero_state = np.zeros(2 ** (2 * system_size), dtype=complex)
for i in range(2 ** system_size):
    state_i = np.zeros(2 ** system_size)
    state_i[i] = 1
    zero_state += np.kron(state_i, state_i)
zero_state = zero_state / np.sqrt(2 ** system_size)
zero_state = np.asmatrix(zero_state).T

target_unitary = scio.loadmat('./exp_ideal_{}_qubit_t=1.mat'.format(system_size))
real_state_tmp = np.matmul(np.kron(target_unitary['exp_ideal'], Identity(system_size)), zero_state)
real_state = real_state_tmp
real_state_m = real_state_tmp@real_state_tmp.getH()
# print(real_state)


step_size = 1


# define generator
qc_gen = Generator(system_size)

#   discriminator
herm = [I, X, Y, Z]
qc_dis = Discriminator(herm, system_size * 2)

if __name__ == '__main__':
    starttime = datetime.now()
    for iter in range(epochs):
        print("==================================================")
        print("Epoch {}, Step_size {}".format(iter + 1, eta))

        # print(compute_cost(qc_gen, qc_dis))
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if iter % step_size == 0:
            # Generator gradient descent
            qc_gen.update_gen(qc_dis)
            print("Loss after generator step: {}".format(compute_cost(qc_gen, qc_dis)))

        # Discriminator gradient ascent
        qc_dis.update_dis(qc_gen)
        print("Loss after discriminator step: {}".format(compute_cost(qc_gen, qc_dis)))

        if iter % 1 == 0:
            cost = compute_cost(qc_gen, qc_dis)
            fidelity = compute_fidelity(qc_gen, zero_state, real_state)
            losses.append(cost)
            fidelities.append(fidelity)

        if iter % 10 ==0:
            endtime = datetime.now()
            training_duration = (endtime - starttime).seconds/np.float(3600)
            param = 'epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n'.format(iter,round(fidelity,6),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),round(training_duration,2))
            train_log(param,'./[log]{}qubit_layer={}.txt'.format(system_size,layer,label))
        print("Fidelity between real and fake state: {}".format(fidelity))
        print("==================================================")

        if (decay):
            eta = (initial_eta * (epochs - iter - 1) +
                   (1e-2 * initial_eta) * iter) / epochs

        param = "{} {}\n".format(fidelity,cost)
        train_log(param, './{}qubit_fidelity&loss.txt'.format(system_size, layer, label))

        save_model(qc_gen, './model/model{}_layer{}_{}_gen.qc'.format(system_size, layer, label))
        save_model(qc_dis, './model/model{}_layer{}_{}_dis.qc'.format(system_size, layer, label))

    fig, (axs,axs2) = plt.subplots()
    axs.plot(range(epochs), fidelities)
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Fidelity between real and fake states')
    axs2.plot(range(epochs), losses)
    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Wasserstein Loss')
    plt.tight_layout()
    plt.savefig('./figure/{}qubit_layer={}_{}_{}.png'.format(system_size, layer, epochs, label))

    print("end")


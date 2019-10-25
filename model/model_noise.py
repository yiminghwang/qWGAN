#!/usr/bin/env python

"""
    model_noise.py: the model of generator and discriminator(noise)

"""
import random
from scipy.linalg import expm, sqrtm
import numpy as np
from config_mixed import *
from tools.qcircuit import Quantum_Gate, Quantum_Circuit
from tools.utils import get_zero_state
from model.model_mixed import Generator,Discriminator

np.random.seed()

def compute_cost(gen, dis, real_state):

    G_list = gen.getGen()

    zero_state = get_zero_state(gen.size)

    P = np.zeros_like(G_list[0])
    for p, g in zip(gen.prob_gen, G_list):
        state_i = np.matmul(g, zero_state)
        P += p * (np.matmul(state_i, state_i.getH()))

    Q = real_state

    psi = dis.getPsi()
    phi = dis.getPhi()

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

    psiterm = np.trace(np.matmul(Q, psi))

    phiterm = np.trace(np.matmul(P, phi))

    term1 = np.trace(np.matmul(A, P)) * np.trace(np.matmul(B, Q))
    term2 = np.trace(np.matmul(A, np.matmul(P, np.matmul(B, Q))))
    term3 = np.trace(np.matmul(P, np.matmul(A, np.matmul(Q, B))))
    term4 = np.trace(np.matmul(B, P)) * np.trace(np.matmul(A, Q))

    regterm = lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)

    return np.real(psiterm - phiterm - regterm)

def compute_fidelity(gen, state, real_state):
    '''
        calculate the fidelity between target state and fake state
    :param gen: generator(Generator)
    :param state: vector(array), input state
    :return:
    '''

    # for density matrix
    G_list = gen.getGen()

    fake_state = np.zeros_like(G_list[0])
    for p, g in zip(gen.prob_gen, G_list):
        state_i = np.matmul(g, state)
        fake_state += p * (np.matmul(state_i, state_i.getH()))

    tmp = sqrtm(fake_state)
    fidelity = sqrtm(np.matmul(tmp, np.matmul(real_state, tmp)))

    return np.real(np.square(np.trace(fidelity)))

def getfake_dens_mat(G, state):
    f_state = np.matmul(G, state)
    f_denmat = np.matmul(f_state, f_state.getH())
    return f_denmat

class Gen:

    def __init__(self, system_size, num_to_mix, mu, sigma):
        self.size = system_size
        self.num_to_mix = num_to_mix
        self.prob_gen = self.init_prob_gen()
        self.qc_list = list()
        self.init_qcircuit()
        self.mu = mu
        self.sigma = sigma

    def set_qcircuit(self, qc_list):
        self.qc_list = qc_list

    def init_qcircuit(self):
        self.qc_list[:] = []
        for i in range(self.num_to_mix):
            qcircuit = Quantum_Circuit(self.size, "generator")
            self.qc_list.append(qcircuit)
        return self.qc_list

    def init_prob_gen(self):
        # return self._softmax()
        return [1,0]

    def getGen(self):

        g_list = list()

        for g in self.qc_list:
            g_list.append(g.get_mat_rep())
        return g_list

    def _grad_theta(self, dis, real_state):

        G_list = self.getGen()

        Q = real_state
        zero_state = get_zero_state(self.size)

        phi = dis.getPhi()
        psi = dis.getPsi()

        grad = list()

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

        for G, j in zip(G_list, range(len(self.qc_list))):

            fake_state = np.matmul(G, zero_state)

            grad_g_psi = list()
            grad_g_phi = list()
            grad_g_reg = list()

            for i in range(self.qc_list[j].depth):

                grad_i = self.qc_list[j].get_grad_mat_rep(i)

                # for psi term
                grad_g_psi.append(0)

                # for phi term
                fake_grad = np.matmul(grad_i, zero_state)
                g_Gi = self.prob_gen[j] * (
                        np.matmul(fake_grad, fake_state.getH()) + np.matmul(fake_state, fake_grad.getH()))

                grad_g_phi.append(np.trace(np.matmul(g_Gi, phi)))

                # for reg term
                term1 = np.trace(np.matmul(A, g_Gi)) * np.trace(np.matmul(B, Q))
                term2 = np.trace(np.matmul(A, np.matmul(g_Gi, np.matmul(B, Q))))
                term3 = np.trace(np.matmul(g_Gi, np.matmul(A, np.matmul(Q, B))))
                term4 = np.trace(np.matmul(B, g_Gi)) * np.trace(np.matmul(A, Q))

                tmp_reg_grad = lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)
                grad_g_reg.append(tmp_reg_grad)

            g_psi = np.asarray(grad_g_psi)
            g_phi = np.asarray(grad_g_phi)
            g_reg = np.asarray(grad_g_reg)

            for i in range(len(g_psi)):
                g_psi[i] += random.gauss(self.mu, self.sigma)
                g_phi[i] += random.gauss(self.mu, self.sigma)
                g_reg[i] += random.gauss(self.mu, self.sigma)

            grad.append(np.real(g_psi - g_phi - g_reg))

        return grad

    def update_gen(self, dis, real_state):

        grad_theta_list = self._grad_theta(dis,real_state)
        new_angle = np.zeros((self.num_to_mix, self.qc_list[0].depth))
        for j, qc in zip(range(self.num_to_mix), self.qc_list):
            # get the new angles of jth circuit
            tmp_angle = list()
            for gate, grad_theta_j in zip(qc.gates, grad_theta_list[j]):
                tmp_angle.append(gate.angle - theta_lr * grad_theta_j)
            new_angle[j] = tmp_angle

        ##update angle
        for j in range(self.num_to_mix):
            for i in range(grad_theta_list[j].size):
                self.qc_list[j].gates[i].angle = new_angle[j][i]


class Dis:

    def __init__(self, herm, system_size, mu, sigma):
        self.size = system_size
        self.herm = herm
        self.alpha = np.zeros((self.size, len(self.herm)))
        self.beta = np.zeros((self.size, len(self.herm)))
        self._init_params()
        self.mu = mu
        self.sigma = sigma

    def _init_params(self):
        # Discriminator Parameters

        for i in range(self.size):
            self.alpha[i] = -1 + 2 * np.random.random(len(self.herm))
            self.beta[i] = -1 + 2 * np.random.random(len(self.herm))

    def getPsi(self):
        """
            get matrix representation of real part of discriminator
        :param alpha:
                    parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        :return:
        """
        psi = 1
        for i in range(self.size):
            psi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
            psi = np.kron(psi, psi_i)
        return psi

    def getPhi(self):
        """
            get matrix representation of fake part of discriminator
        :param:
                    parameters of psi(ndarray):size = [num_qubit, 4]
                    0: I
                    1: X
                    2: Y
                    3: Z
        :return:
        """
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

    def _grad_alpha(self, gen, real_state):
        G_list = gen.getGen()

        psi = self.getPsi()
        phi = self.getPhi()

        zero_state = get_zero_state(self.size)

        P = np.zeros_like(G_list[0])
        for p, g in zip(gen.prob_gen, G_list):
            state_i = np.matmul(g, zero_state)
            P += p * (np.matmul(state_i, state_i.getH()))

        Q = real_state

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

                gradpsi_list.append(np.trace(np.matmul(Q, grad_psi)))

                gradphi_list.append(0)

                tmp_grad_psi = (1 / lamb) * np.matmul(grad_psi, B)

                term1 = np.trace(np.matmul(A, P)) * np.trace(np.matmul(tmp_grad_psi, Q))
                term2 = np.trace(np.matmul(A, np.matmul(P, np.matmul(tmp_grad_psi, Q))))
                term3 = np.trace(np.matmul(P, np.matmul(A, np.matmul(Q, tmp_grad_psi))))
                term4 = np.trace(np.matmul(tmp_grad_psi, P)) * np.trace(np.matmul(A, Q))

                gradreg_list.append(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4))

            # calculate grad of psi term
            grad_psi_term[:, type] += np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] += np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] += np.asarray(gradreg_list)

            for i in range(len(gradpsi)):
                grad_psi_term[i,type] += random.gauss(self.mu, self.sigma)
                grad_phi_term[i,type] += random.gauss(self.mu, self.sigma)
                grad_reg_term[i,type] += random.gauss(self.mu, self.sigma)

        return np.real(grad_psi_term - grad_phi_term - grad_reg_term)

    def _grad_beta(self, gen, real_state):

        G_list = gen.getGen()

        psi = self.getPsi()
        phi = self.getPhi()

        zero_state = get_zero_state(self.size)

        P = np.zeros_like(G_list[0])
        for p, g in zip(gen.prob_gen, G_list):
            state_i = np.matmul(g, zero_state)
            P += p * (np.matmul(state_i, state_i.getH()))

        Q = real_state

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

                gradphi_list.append(np.trace(np.matmul(P, grad_phi)))

                tmp_grad_phi = -1 / lamb * np.matmul(grad_phi, A)

                term1 = np.trace(np.matmul(tmp_grad_phi, P)) * np.trace(np.matmul(B, Q))
                term2 = np.trace(np.matmul(tmp_grad_phi, np.matmul(P, np.matmul(B, Q))))
                term3 = np.trace(np.matmul(P, np.matmul(tmp_grad_phi, np.matmul(Q, B))))
                term4 = np.trace(np.matmul(B, P)) * np.trace(np.matmul(tmp_grad_phi, Q))

                gradreg_list.append(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4))

            # calculate grad of psi term
            grad_psi_term[:, type] += np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] += np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] += np.asarray(gradreg_list)

            for i in range(len(gradphi)):
                grad_psi_term[i, type] += random.gauss(self.mu, self.sigma)
                grad_phi_term[i, type] += random.gauss(self.mu, self.sigma)
                grad_reg_term[i, type] += random.gauss(self.mu, self.sigma)

        return np.real(grad_psi_term - grad_phi_term - grad_reg_term)

    def update_dis(self, gen, real_state):

        # update alpha
        new_alpha = self.alpha + psi_lr * self._grad_alpha(gen,real_state)

        # update beta
        new_beta = self.beta + phi_lr * self._grad_beta(gen,real_state)

        self.alpha = new_alpha
        self.beta = new_beta
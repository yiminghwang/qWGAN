#!/usr/bin/env python
"""
    model_hs.py: the model of generator and discriminator(hamiltonian simulation)

"""

from scipy.linalg import expm, sqrtm
import numpy as np
from config_hs import *
from momentum import MomentumOptimizer
from tools.qcircuit import Quantum_Gate, Quantum_Circuit, Identity
from tools.utils import get_zero_state,get_maximally_entangled_state

np.random.seed()

def compute_cost(gen, dis, real_state):

    G = gen.getGen()
    psi = dis.getPsi()
    phi = dis.getPhi()

    fake_state = np.matmul(G, input_state)

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

    psiterm = np.trace(np.matmul(np.matmul(real_state, real_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(fake_state, fake_state.getH()), phi))

    regterm = np.asscalar(
        lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8))

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(gen, input_state, real_state, type='training'):
    '''
        calculate the fidelity between target state and fake state
    :param gen: generator(Generator)
    :param state: vector(array), input state
    :return:
    '''
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

    def set_qcircuit(self, qc):
        self.qc = qc

    def init_qcircuit(self):
        qcircuit = Quantum_Circuit(self.size, "generator")
        return qcircuit

    def getGen(self):
        return np.kron(self.qc.get_mat_rep(), Identity(self.size))

    def _grad_theta(self, dis, real_state):

        G = self.getGen()

        phi = dis.getPhi()
        psi = dis.getPsi()

        fake_state = np.matmul(G, input_state)

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

        grad_g_psi = list()
        grad_g_phi = list()
        grad_g_reg = list()

        for i in range(self.qc.depth):

            grad_i = np.kron(self.qc.get_grad_mat_rep(i), Identity(system_size))
            # for psi term
            grad_g_psi.append(0)

            # for phi term
            fake_grad = np.matmul(grad_i, input_state)
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

        return grad

    def update_gen(self, dis, real_state):

        theta = []
        for gate in self.qc.gates:
            theta.append(gate.angle)

        grad = np.asarray(self._grad_theta(dis, real_state))
        theta = np.asarray(theta)
        new_angle = self.optimizer.compute_grad(theta,grad,'min')
        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]


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

    def _grad_alpha(self, gen, real_state):
        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, input_state)

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

    def _grad_beta(self, gen, real_state):

        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        fake_state = np.matmul(G, input_state)

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

        return grad

    def update_dis(self, gen, real_state):

        grad_alpha = self._grad_alpha(gen, real_state)
        # update alpha
        new_alpha = self.optimizer_psi.compute_grad(self.alpha, grad_alpha, 'max')
        # new_alpha = self.alpha + eta * self._grad_alpha(gen)

        grad_beta = self._grad_beta(gen, real_state)
        # update beta
        new_beta = self.optimizer_phi.compute_grad(self.beta, grad_beta, 'max')
        # new_beta = self.beta + eta * self._grad_beta(gen)

        self.alpha = new_alpha
        self.beta = new_beta


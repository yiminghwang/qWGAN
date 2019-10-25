#!/usr/bin/env python

"""
    model_pure.py: the model of generator and discriminator(pure states)

"""

from scipy.linalg import expm
import numpy as np
from config_pure import *
from tools.qcircuit import Quantum_Gate, Quantum_Circuit
from tools.utils import get_zero_state

np.random.seed()

def compute_cost(gen, dis, real_state):
    '''
        calculate the loss
    :param gen: generator(Generator)
    :param dis: discriminator(Discriminator)
    :return:
    '''

    G = gen.getGen()
    psi = dis.getPsi()
    phi = dis.getPhi()

    zero_state = get_zero_state(gen.size)

    fake_state = np.matmul(G ,zero_state)

    try:
        A = expm(np.float(-1 / lamb) * phi)
    except Exception:
        print('cost function -1/lamb:\n',(-1/lamb))
        print('size of phi:\n',phi.shape)

    try:
        B = expm(np.float(1 / lamb) * psi)
    except Exception:
        print('cost function 1/lamb:\n',(1/lamb))
        print('size of psi:\n',psi.shape)

    term1 = np.matmul(fake_state.getH(),np.matmul(A,fake_state))
    term2 = np.matmul(real_state.getH(),np.matmul(B,real_state))

    term3 = np.matmul(fake_state.getH(),np.matmul(B,real_state))
    term4 = np.matmul(real_state.getH(),np.matmul(A,fake_state))

    term5 = np.matmul(fake_state.getH(),np.matmul(A,real_state))
    term6 = np.matmul(real_state.getH(),np.matmul(B,fake_state))

    term7 = np.matmul(fake_state.getH(),np.matmul(B,fake_state))
    term8 = np.matmul(real_state.getH(),np.matmul(A,real_state))

    psiterm = np.asscalar(np.matmul(real_state.getH(), np.matmul(psi , real_state)))
    phiterm = np.asscalar(np.matmul(fake_state.getH(), np.matmul(phi , fake_state)))


    regterm = np.asscalar(
        lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8))

    return np.real(psiterm - phiterm - regterm)


def compute_fidelity(gen, state, real_state):
    '''
        calculate the fidelity between target state and fake state
    :param gen:   generator(Generator)
    :param state: vector(array), input state
    :return:
    '''
    G = gen.getGen()
    fake_state = np.matmul(G , state)
    return np.abs(np.asscalar(np.matmul(real_state.getH() , fake_state))) ** 2


class Generator:
    def __init__(self, system_size):
        self.size = system_size
        self.qc = self.init_qcircuit()

    def reset_angles(self):
        theta = np.random.random(len(self.qc.gates))
        for i in range(len(self.qc.gates)):
            self.qc.gates[i].angle = theta[i]

    def init_qcircuit(self):
        qcircuit = Quantum_Circuit(self.size, "generator")
        return qcircuit

    def set_qcircuit(self, qc):
        self.qc = qc

    def getGen(self):
        return self.qc.get_mat_rep()

    def _grad_theta(self, dis, real_state):

        G = self.getGen()

        phi = dis.getPhi()
        psi = dis.getPsi()

        zero_state = get_zero_state(self.size)

        fake_state = np.matmul(G , zero_state)

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

        grad_g = list()

        grad_g_psi = list()
        grad_g_phi = list()
        grad_g_reg = list()

        for i in range(self.qc.depth):
            grad_g.append(self.qc.get_grad_mat_rep(i))

        for grad_i in grad_g:
            # for psi term
            grad_g_psi.append(0)

            # for phi term
            fake_grad = np.matmul(grad_i , zero_state)
            tmp_grad = np.matmul(fake_grad.getH() , np.matmul(phi , fake_state)) + np.matmul(fake_state.getH() ,np.matmul(phi , fake_grad))

            grad_g_phi.append(np.asscalar(tmp_grad))

            # for reg term
            term1 = np.matmul(fake_grad.getH() , np.matmul(A , fake_state)) * np.matmul(real_state.getH() , np.matmul(B , real_state))
            term2 = np.matmul(fake_state.getH() , np.matmul(A , fake_grad)) * np.matmul(real_state.getH() , np.matmul(B , real_state))

            term3 = np.matmul(fake_grad.getH() , np.matmul(B , real_state)) * np.matmul(real_state.getH() ,np.matmul( A , fake_state))
            term4 = np.matmul(fake_state.getH() , np.matmul(B , real_state)) * np.matmul(real_state.getH() , np.matmul(A , fake_grad))

            term5 = np.matmul(fake_grad.getH() , np.matmul(A , real_state)) * np.matmul(real_state.getH() , np.matmul(B , fake_state))
            term6 = np.matmul(fake_state.getH() , np.matmul(A , real_state)) * np.matmul(real_state.getH() , np.matmul(B , fake_grad))

            term7 = np.matmul(fake_grad.getH() , np.matmul(B , fake_state)) * np.matmul(real_state.getH() , np.matmul(A , real_state))
            term8 = np.matmul(fake_state.getH() , np.matmul(B , fake_grad)) * np.matmul(real_state.getH() , np.matmul(A , real_state))

            tmp_reg_grad = lamb / np.e * (
                    cst1 * (term1 + term2) - cst2 * (term3 + term4) - cst2 * (term5 + term6) + cst3 * (term7 + term8))

            grad_g_reg.append(np.asscalar(tmp_reg_grad))

        g_psi = np.asarray(grad_g_psi)
        g_phi = np.asarray(grad_g_phi)
        g_reg = np.asarray(grad_g_reg)

        grad = np.real(g_psi - g_phi - g_reg)

        return grad

    def update_gen(self, dis,real_state):
        new_angle = []

        grad_list = self._grad_theta(dis,real_state)
        for gate, grad in zip(self.qc.gates, grad_list):
            new_angle.append(gate.angle - eta * grad)

        for i in range(self.qc.depth):
            self.qc.gates[i].angle = new_angle[i]


class Discriminator:

    def __init__(self, herm, system_size):
        self.size = system_size
        self.herm = herm
        self.alpha = np.zeros((self.size, len(self.herm)))
        self.beta = np.zeros((self.size, len(self.herm)))
        self._init_params()

    def _init_params(self):
        # initial Discriminator Parameters

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

    def _grad_alpha(self, gen,real_state):
        G = gen.getGen()
        psi = self.getPsi()
        phi = self.getPhi()

        zero_state = get_zero_state(self.size)
        fake_state = np.matmul(G , zero_state)

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
                gradpsi_list.append(np.asscalar(np.matmul(real_state.getH() , np.matmul(grad_psi , real_state))))

                gradphi_list.append(0)

                term1 = cs * np.matmul(fake_state.getH() , np.matmul(A , fake_state)) * np.matmul(real_state.getH() , np.matmul(grad_psi,np.matmul(B , real_state)))
                term2 = cs * np.matmul(fake_state.getH() , np.matmul(grad_psi , np.matmul(B , real_state))) * np.matmul(real_state.getH() ,np.matmul( A , fake_state))
                term3 = cs * np.matmul(fake_state.getH() ,np.matmul(A , real_state))* np.matmul(real_state.getH() ,np.matmul( grad_psi,np.matmul(B , fake_state)))
                term4 = cs * np.matmul(fake_state.getH() ,np.matmul(grad_psi ,np.matmul( B , fake_state))) * np.matmul(real_state.getH() ,np.matmul( A , real_state))

                gradreg_list.append(
                    np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        # print("grad_alpha:\n",np.real(grad_psi_term - grad_phi_term - grad_reg_term))
        return np.real(grad_psi_term - grad_phi_term - grad_reg_term)

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

        zero_state = get_zero_state(self.size)
        fake_state = np.matmul(G , zero_state)

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

                gradphi_list.append(np.asscalar(np.matmul(fake_state.getH() ,np.matmul( grad_phi , fake_state))))

                term1 = cs * np.matmul(fake_state.getH() ,np.matmul( grad_phi ,np.matmul( A , fake_state))) * np.matmul(real_state.getH() ,np.matmul( B , real_state))
                term2 = cs * np.matmul(fake_state.getH() , np.matmul(B , real_state)) * np.matmul(real_state.getH() ,np.matmul( grad_phi ,np.matmul( A , fake_state)))
                term3 = cs * np.matmul(fake_state.getH() ,np.matmul( grad_phi ,np.matmul( A , real_state))) * np.matmul(real_state.getH() ,np.matmul( B , fake_state))
                term4 = cs * np.matmul(fake_state.getH() ,np.matmul(B , fake_state)) * np.matmul(real_state.getH() ,np.matmul( grad_phi ,np.matmul( A , real_state)))

                gradreg_list.append(
                    np.asscalar(lamb / np.e * (cst1 * term1 - cst2 * term2 - cst2 * term3 + cst3 * term4)))

            # calculate grad of psi term
            grad_psi_term[:, type] = np.asarray(gradpsi_list)

            # calculate grad of phi term
            grad_phi_term[:, type] = np.asarray(gradphi_list)

            # calculate grad of reg term
            grad_reg_term[:, type] = np.asarray(gradreg_list)

        # print("grad_beta:\n",np.real(grad_psi_term - grad_phi_term - grad_reg_term))
        return np.real(grad_psi_term - grad_phi_term - grad_reg_term)

    def update_dis(self, gen,real_state):

        # update alpha
        new_alpha = self.alpha + eta * self._grad_alpha(gen,real_state)

        # update beta
        new_beta = self.beta + eta * self._grad_beta(gen,real_state)

        self.alpha = new_alpha
        self.beta = new_beta
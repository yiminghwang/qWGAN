#!/usr/bin/env python

"""
    qcircuit.py: including base components and definition of quantum circuit simulation.

"""
import traceback

import numpy as np
import scipy.linalg as linalg
import os
import random

import sys
from scipy.sparse import dok_matrix

I = np.eye(2)

# Pauli matrices
X = np.matrix([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = np.matrix([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = np.matrix([[1, 0], [0, -1]])  #: Pauli-Z matrix
Hadamard = np.matrix([[1, 1], [1, -1]] / np.sqrt(2))  #: Hadamard gate

zero = np.matrix([[1, 0], [0, 0]])
one = np.matrix([[0, 0], [0, 1]])

# Two qubit gates
CNOT = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [
    0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [
    0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
CZ = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [
    0, 0, 1, 0], [0, 0, 0, -1]])  #: CZ gate

global param_table
param_table = dict()


def Identity(size):
    matrix = 1
    for i in range(1, size + 1):
        matrix = np.kron(matrix, I)
    return matrix

def CSWAP(size):
    '''
        get control swap gate
    :param size:
    :return:
    '''
    dim = 2 * size
    C_SWAP = dok_matrix((2**(dim+1),2**(dim+1)))

    dim1 = 2 ** size
    SWAP = dok_matrix((dim1 * dim1, dim1 * dim1))

    for i in range(2**dim):
        C_SWAP[i,i] = 1

    for i in range(dim1):
        for j in range(dim1):
            SWAP[i * dim1 + j, j * dim1 + i] = 1
            SWAP[j * dim1 + i, i * dim1 + j] = 1
            C_SWAP[i * dim1 + j + 2**dim,j * dim1 + i + 2**dim] = 1
            C_SWAP[j * dim1 + i + 2**dim,i * dim1 + j + 2**dim] = 1
    # C_SWAP[SWAP.nonzero()] = SWAP[SWAP.nonzero()]
    return C_SWAP - np.zeros((2 ** (dim + 1), 2 ** (dim + 1))),SWAP

def CSWAP_T(size):
    '''
        get control swap gate
    :param size:
    :return:
    '''
    dim = 2 * size
    C_SWAP = dok_matrix((2**(dim+1),2**(dim+1)))

    dim1 = 2 ** size
    SWAP = dok_matrix((dim1 * dim1, dim1 * dim1))
    # C_SWAP = np.zeros((2 ** (dim + 1), 2 ** (dim + 1)))
    # SWAP = np.zeros((dim * dim, dim * dim))

    for i in range(dim1):
        for j in range(dim1):
            SWAP[i * dim1 + j,j * dim1 + i] = 1
            SWAP[j * dim1 + i,i * dim1 + j] = 1

    C_SWAP[SWAP.nonzero()] = SWAP[SWAP.nonzero()]

    for i in range(2**dim,2**(dim+1)):
        C_SWAP[i,i] = 1

    return C_SWAP - np.zeros((2 ** (dim + 1), 2 ** (dim + 1))),SWAP

def mCNOT(size, control, target):
    gate = np.asarray(X)
    U = expan_2qubit_gate(gate,size,control,target)
    return U

def expan_2qubit_gate(gate,size,control,target):
    wires = np.asarray((control,target))

    if control > size - 1:
        raise IndexError('index is out of bound of wires')
    if target > size - 1:
        raise IndexError('index is out of bound of wires')
    if control - target == 0:
        raise IndexError('index should not be same')

    a = np.min(wires)
    b = np.max(wires)
    if a == control:
        U_one = np.kron(Identity(control), np.kron(zero, Identity(size - control - 1)))
        between = b-a-1
        U_two = np.kron(Identity(control),np.kron(one, np.kron(Identity(between), np.kron(gate, Identity(size - target - 1)))))
    else:
        U_one = np.kron(Identity(control), np.kron(zero, Identity(size - control - 1)))
        between = a-b-1
        U_two = np.kron(Identity(target),np.kron(gate,np.kron(Identity(between),np.kron(one,Identity(size-control-1)))))
    return U_one+U_two


def XX_Rotation1(size, qubit1, qubit2, param, is_grad):
    U = expan_2qubit_gate(linalg.expm(-1J * param * np.kron(X, X)),size,qubit1,qubit2)
    return U

def YY_Rotation1(size, qubit1, qubit2, param, is_grad):
    U = expan_2qubit_gate(linalg.expm(-1J * param * np.kron(Y, Y)),size,qubit1,qubit2)
    return U

def ZZ_Rotation1(size, qubit1, qubit2, param, is_grad):
    U = expan_2qubit_gate(linalg.expm(1J/2 * param * np.kron(Z, Z)),size,qubit1,qubit2)
    return U

def XX_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)

    if is_grad == False:
        try:
            return linalg.expm(-1J * param * matrix)
            # return matrix
        except Exception:
            print('param:\n:',param)
    else:
        return -1J * np.matmul(matrix, linalg.expm(-1J * param * matrix))

def YY_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Y)
        else:
            matrix = np.kron(matrix, I)

    if is_grad == False:
        try:
            return linalg.expm(-1J * param * matrix)
            # return matrix
        except Exception:
            print('param:\n:',param)
    else:
        return -1J * np.matmul(matrix, linalg.expm(-1J * param * matrix))

def ZZ_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)

    if is_grad == False:
        try:
            return linalg.expm(1J/2 * param * matrix)
            # return -1/2 * matrix
        except Exception:
            print('param:\n:',param)
    else:
        return 1J/2 * np.matmul(matrix, linalg.expm(1J/2 * param * matrix))


def X_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad == False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1J / 2 * param * X))
                except Exception:
                    print('param:\n:', param)
            else:
                matrix = np.kron(matrix, -1J / 2 * X * linalg.expm(-1J / 2 * param * X))
        else:
            matrix = np.kron(matrix, I)

    return matrix

def Y_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad == False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1J / 2 * param * Y))
                except Exception:
                    print('param:\n:', param)
            else:
                matrix = np.kron(matrix, -1J / 2 * Y * linalg.expm(-1J / 2 * param * Y))
        else:
            matrix = np.kron(matrix, I)

    return matrix


def Z_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad == False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1J / 2 * param * Z))
                except Exception:
                    print('param:\n:', param)
            else:
                matrix = np.kron(matrix, -1J / 2 * Z * linalg.expm(-1J / 2 * param * Z))
        else:
            matrix = np.kron(matrix, I)

    return matrix

def Global_phase(size, param, is_grad):
    matrix = np.eye(2**size)
    eA = np.exp(-1J * param**2) * matrix
    if is_grad == False:
        return eA
    else:
        return -1J *2 * param * np.matmul(matrix,eA)

class Quantum_Gate:
    def __init__(self, name, qubit1=None, qubit2=None, **kwarg):
        self.name = name
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.r = self.get_r()
        self.s = self.get_s()

        if "angle" in kwarg:
            self.angle = kwarg["angle"]
        else:
            self.angle = None

    def get_r(self):
        if self.name == 'X' or self.name == 'Y' or self.name == 'Z' or self.name == 'ZZ':
            return 1/2
        elif self.name == 'XX' or self.name == 'YY':
            return 1
        else:
            return None

    def get_s(self):
        if self.r != None:
            return np.pi / (4 * self.r)
        else:
            return None

    def matrix_representation(self, size, is_grad):

        if self.angle != None:
            try:
                param = float(self.angle)
            except:
                param = param_table[self.angle]

        if self.name == "XX":
            return XX_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif self.name == "YY":
            return YY_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif self.name == "ZZ":
            return ZZ_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif (self.name == "Z"):
            return Z_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "X"):
            return X_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "Y"):
            return Y_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "CNOT"):
            return mCNOT(size, self.qubit1, self.qubit2)

        elif (self.name == "G"):
            return Global_phase(size, param, is_grad)
        else:
            raise ValueError("Gate is not defined")

    def matrix_representation_shift_phase(self, size, is_grad, signal):

        if self.angle != None:
            try:
                if self.name == 'G':
                    param = float(self.angle)
                else:
                    param = float(self.angle)
                    if is_grad == True:
                        if signal == '+':
                            param = param + self.s
                        else:
                            param = param - self.s
                        is_grad = False
            except:
                param = param_table[self.angle]

        if self.name == "XX":
            return XX_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif self.name == "YY":
            return YY_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif self.name == "ZZ":
            return ZZ_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        elif (self.name == "Z"):
            return Z_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "X"):
            return X_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "Y"):
            return Y_Rotation(size, self.qubit1, param, is_grad)

        elif (self.name == "G"):
            return Global_phase(size, param, is_grad)

        elif (self.name == "CNOT"):
            return mCNOT(size, self.qubit1, self.qubit2)

        else:
            raise ValueError("Gate is not defined")


class Quantum_Circuit:

    def __init__(self, size, name):
        self.size = size
        self.depth = 0
        self.gates = []
        self.name = name

    def check_ciruit(self):
        for j,gate in zip(range(len(self.gates)),self.gates):
            if gate.qubit1!=None and gate.qubit2!=None:
                if gate.qubit1>self.size-1:
                    print('Error: #{} gate:{} 1qubit is out of range'.format(j, gate.name))
                    os._exit(0)
                elif gate.qubit2>self.size-1:
                    print('Error: #{} gate:{} 2qubit is out of range'.format(j, gate.name))
                    os._exit(0)

    def get_mat_rep(self):
        matrix = Identity(self.size)
        for gate in self.gates:
            g = gate.matrix_representation(self.size, False)
            matrix = np.matmul(g, matrix)
        return np.asmatrix(matrix)

    def get_grad_mat_rep(self, index, signal='none', type='matrix_multiplication'):
        '''
            matrix multipliction: explicit way to calculate the gradient using matrix multiplication
            shift_phase: generate two quantum circuit to calculate the gradient
            Evaluating analytic gradients on quantum hardware
            https://arxiv.org/pdf/1811.11184.pdf
        :param index:
        :param type: the type of calculate gradient
        :return:
        '''
        if type == 'shift_phase':
            matrix = Identity(self.size)
            for j, gate in zip(range(len(self.gates)), self.gates):
                if index == j:
                    g = gate.matrix_representation_shift_phase(self.size, True, signal)
                    matrix = np.matmul(g, matrix)
                else:
                    g = gate.matrix_representation_shift_phase(self.size, False, signal)
                    matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

        elif type == 'matrix_multiplication':
            matrix = Identity(self.size)
            for j, gate in zip(range(len(self.gates)), self.gates):
                if index == j:
                    g = gate.matrix_representation(self.size, True)
                    matrix = np.matmul(g, matrix)
                else:
                    g = gate.matrix_representation(self.size, False)
                    matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

    def get_grad_qc(self,indx,type='0'):
        qc_list = list()
        for j,gate in zip(range(len(self.gates)),self.gates):
            tmp = Quantum_Gate(' ',qubit1=None,qubit2=None,angle=None)
            tmp.name = gate.name
            tmp.qubit1 = gate.qubit1
            tmp.qubit2 = gate.qubit2
            tmp.angle = gate.angle
            if j == indx:
                try:
                    if self.gates[j].name != 'G' or self.gates[j].name !='CNOT':
                        if type == '+':
                            tmp.angle = gate.angle + gate.s
                        elif type == '-':
                            tmp.angle = gate.angle - gate.s
                except:
                    print('param value error')
                qc_list.append(tmp)
            else:
                qc_list.append(tmp)
        return qc_list

    def add_gate(self, quantum_gate):
        self.depth += 1
        self.gates.append(quantum_gate)

# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import unittest

import numpy as np
import scipy

from test.common import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.aqua.input import SVMInput
from qiskit.aqua import run_algorithm, QuantumInstance, aqua_globals, Operator
from qiskit.aqua.algorithms import QSVMVariational
from qiskit.aqua.components.optimizers import SPSA, COBYLA, L_BFGS_B
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ, RY
from qiskit.aqua.components.gradients import *
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.algorithms import VQE



class TestQGradients(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        self.svm_accuracy_threshold = 0.85
        self.vqe_oracle = -1.85727503
        self.vqe_pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }


    def test_svm_variational_objectivefunc_gradient(self):
        n = 2  # dimension of each data point
        seed = 1024
        np.random.seed(seed)
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=20,
                                                                             test_size=10,
                                                                             n=n, gap=0.3)

        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = 2
        optimizer = L_BFGS_B(maxfun=1000)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)
        gradient = ObjectiveFuncGradient() # explicitly create/pass an ObjectiveFuncGradient instance
        svm = QSVMVariational(optimizer, feature_map, var_form, training_input, test_input, gradient=gradient)
        aqua_globals.random_seed = seed
        quantum_instance = QuantumInstance(backend, shots=1024, seed=seed)
        result = svm.run(quantum_instance)
        self.assertGreater(result['testing_accuracy'], self.svm_accuracy_threshold)

    def test_svm_variational_circuits_gradient(self):
        n = 2  # dimension of each data point
        seed = 1024
        np.random.seed(seed)
        sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=20,
                                                                             test_size=10,
                                                                             n=n, gap=0.3)

        backend = BasicAer.get_backend('statevector_simulator')

        num_qubits = 2
        optimizer = L_BFGS_B(maxfun=1000)
        feature_map = SecondOrderExpansion(num_qubits=num_qubits, depth=2)
        var_form = RYRZ(num_qubits=num_qubits, depth=3)
        gradient = CircuitsGradient()
        svm = QSVMVariational(optimizer, feature_map, var_form, training_input, test_input, gradient=gradient)
        aqua_globals.random_seed = seed
        quantum_instance = QuantumInstance(backend, shots=1024, seed=seed)
        result = svm.run(quantum_instance)
        self.assertGreater(result['testing_accuracy'], self.svm_accuracy_threshold)

    def test_vqe_objectivefunc_gradient(self):
        np.random.seed(50)
        qubit_op = Operator.load_from_dict(self.vqe_pauli_dict)
        algo_input = EnergyInput(qubit_op)

        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 3, initial_state=init_state)
        optimizer = L_BFGS_B()
        gradient = ObjectiveFuncGradient()
        algo = VQE(algo_input.qubit_op, var_form, optimizer, 'paulis', gradient=gradient)
        quantum_instance = QuantumInstance(backend)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result['energy'], self.vqe_oracle, places=5)

    def test_vqe_operator_gradient(self):
        np.random.seed(50)
        qubit_op = Operator.load_from_dict(self.vqe_pauli_dict)
        algo_input = EnergyInput(qubit_op)

        backend = BasicAer.get_backend('statevector_simulator')
        num_qubits = algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 3, initial_state=init_state)
        optimizer = L_BFGS_B()
        gradient = OperatorGradient()

        algo = VQE(algo_input.qubit_op, var_form, optimizer, 'paulis', gradient=gradient)
        quantum_instance = QuantumInstance(backend)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result['energy'], self.vqe_oracle, places=5)

def ad_hoc_data(training_size, test_size, n, gap):
    class_labels = [r'A', r'B']
    if n == 2:
        N = 100
    elif n == 3:
        N = 20   # courseness of data seperation

    label_train = np.zeros(2*(training_size+test_size))
    sample_train = []
    sampleA = [[0 for x in range(n)] for y in range(training_size+test_size)]
    sampleB = [[0 for x in range(n)] for y in range(training_size+test_size)]

    sample_Total = [[[0 for x in range(N)] for y in range(N)] for z in range(N)]

    interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2*np.pi/N

    sx = np.array([[0, 1], [1, 0]])
    X = np.asmatrix(sx)
    sy = np.array([[0, -1j], [1j, 0]])
    Y = np.asmatrix(sy)
    sz = np.array([[1, 0], [0, -1]])
    Z = np.asmatrix(sz)
    J = np.array([[1, 0], [0, 1]])
    J = np.asmatrix(J)
    H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    H2 = np.kron(H, H)
    H3 = np.kron(H, H2)
    H = np.asmatrix(H)
    H2 = np.asmatrix(H2)
    H3 = np.asmatrix(H3)

    f = np.arange(2**n)

    my_array = [[0 for x in range(n)] for y in range(2**n)]

    for arindex in range(len(my_array)):
        temp_f = bin(f[arindex])[2:].zfill(n)
        for findex in range(n):
            my_array[arindex][findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1)**(2*my_array.sum(axis=0) > n)
    parity = (-1)**(my_array.sum(axis=0))
    dict1 = (-1)**(my_array[0])
    if n == 2:
        D = np.diag(parity)
    elif n == 3:
        D = np.diag(maj)

    Basis = np.random.random((2**n, 2**n)) + 1j*np.random.random((2**n, 2**n))
    Basis = np.asmatrix(Basis).getH()*np.asmatrix(Basis)

    [S, U] = np.linalg.eig(Basis)

    idx = S.argsort()[::-1]
    S = S[idx]
    U = U[:, idx]

    M = (np.asmatrix(U)).getH()*np.asmatrix(D)*np.asmatrix(U)

    psi_plus = np.transpose(np.ones(2))/np.sqrt(2)
    psi_0 = 1
    for k in range(n):
        psi_0 = np.kron(np.asmatrix(psi_0), np.asmatrix(psi_plus))

    sample_total_A = []
    sample_total_B = []
    sample_total_void = []
    if n == 2:
        for n1 in range(N):
            for n2 in range(N):
                x1 = steps*n1
                x2 = steps*n2
                phi = x1*np.kron(Z, J) + x2*np.kron(J, Z) + (np.pi-x1)*(np.pi-x2)*np.kron(Z, Z)
                Uu = scipy.linalg.expm(1j*phi)
                psi = np.asmatrix(Uu)*H2*np.asmatrix(Uu)*np.transpose(psi_0)
                temp = np.asscalar(np.real(psi.getH()*M*psi))
                if temp > gap:
                    sample_Total[n1][n2] = +1
                elif temp < -gap:
                    sample_Total[n1][n2] = -1
                else:
                    sample_Total[n1][n2] = 0

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == +1:
                sampleA[tr] = [2*np.pi*draw1/N, 2*np.pi*draw2/N]
                tr += 1

        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            if sample_Total[draw1][draw2] == -1:
                sampleB[tr] = [2*np.pi*draw1/N, 2*np.pi*draw2/N]
                tr += 1

        sample_train = [sampleA, sampleB]

        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}



    elif n == 3:
        for n1 in range(N):
            for n2 in range(N):
                for n3 in range(N):
                    x1 = steps*n1
                    x2 = steps*n2
                    x3 = steps*n3
                    phi = x1*np.kron(np.kron(Z, J), J) + x2*np.kron(np.kron(J, Z), J) + x3*np.kron(np.kron(J, J), Z) + \
                        (np.pi-x1)*(np.pi-x2)*np.kron(np.kron(Z, Z), J)+(np.pi-x2)*(np.pi-x3)*np.kron(np.kron(J, Z), Z) + \
                        (np.pi-x1)*(np.pi-x3)*np.kron(np.kron(Z, J), Z)
                    Uu = scipy.linalg.expm(1j*phi)
                    psi = np.asmatrix(Uu)*H3*np.asmatrix(Uu)*np.transpose(psi_0)
                    temp = np.asscalar(np.real(psi.getH()*M*psi))
                    if temp > gap:
                        sample_Total[n1][n2][n3] = +1
                        sample_total_A.append([n1, n2, n3])
                    elif temp < -gap:
                        sample_Total[n1][n2][n3] = -1
                        sample_total_B.append([n1, n2, n3])
                    else:
                        sample_Total[n1][n2][n3] = 0
                        sample_total_void.append([n1, n2, n3])

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == +1:
                sampleA[tr] = [2*np.pi*draw1/N, 2*np.pi*draw2/N, 2*np.pi*draw3/N]
                tr += 1

        tr = 0
        while tr < (training_size+test_size):
            draw1 = np.random.choice(N)
            draw2 = np.random.choice(N)
            draw3 = np.random.choice(N)
            if sample_Total[draw1][draw2][draw3] == -1:
                sampleB[tr] = [2*np.pi*draw1/N, 2*np.pi*draw2/N, 2*np.pi*draw3/N]
                tr += 1

        sample_train = [sampleA, sampleB]

        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}


    return sample_Total, training_input, test_input, class_labels

if __name__ == '__main__':
    unittest.main()

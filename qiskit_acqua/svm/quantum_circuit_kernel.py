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

import itertools

from qiskit import QuantumCircuit, QuantumProgram, QuantumRegister, ClassicalRegister
import numpy as np

np.warnings.filterwarnings('ignore')


def entangler_map_creator(n):
    if n == 2:
        entangler_map = {0: [1]}
    elif n == 3:
        entangler_map = {0: [2, 1],
                         1: [2]}
    elif n == 4:
        entangler_map = {0: [2, 1],
                         1: [2],
                         3: [2]}
    elif n == 5:
        entangler_map = {0: [2, 1],
                         1: [2],
                         3: [2, 4],
                         4: [2]}
    return entangler_map



def inner_prod_circuit_ML(entangler_map, coupling_map, initial_layout,n, x_vec1, x_vec2, meas_string = None, measurement = True):

    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
    trial_circuit = QuantumCircuit(q, c)

    #write input state from sample distribution
    for r in range(len(x_vec1)):
        trial_circuit.h(q[r])
        trial_circuit.u1(2*x_vec1[r], q[r])
    for node in entangler_map:
        for j in entangler_map[node]:
            trial_circuit.cx(q[node], q[j])
            trial_circuit.u1(2*(np.pi-x_vec1[node])*(np.pi-x_vec1[j]), q[j])
            trial_circuit.cx(q[node], q[j])

    for r in range(len(x_vec1)):
        trial_circuit.h(q[r])
        trial_circuit.u1(2*x_vec1[r], q[r])
    for node in entangler_map:
        for j in entangler_map[node]:
            trial_circuit.cx(q[node], q[j])
            trial_circuit.u1(2*(np.pi-x_vec1[node])*(np.pi-x_vec1[j]), q[j])
            trial_circuit.cx(q[node], q[j])


    for node in entangler_map:
        for j in entangler_map[node]:
            trial_circuit.cx(q[node], q[j])
            trial_circuit.u1(-2*(np.pi-x_vec2[node])*(np.pi-x_vec2[j]), q[j])
            trial_circuit.cx(q[node], q[j])
    for r in range(len(x_vec2)):
        trial_circuit.u1(-2*x_vec2[r], q[r])
        trial_circuit.h(q[r])

    for node in entangler_map:
        for j in entangler_map[node]:
            trial_circuit.cx(q[node], q[j])
            trial_circuit.u1(-2*(np.pi-x_vec2[node])*(np.pi-x_vec2[j]), q[j])
            trial_circuit.cx(q[node], q[j])
    for r in range(len(x_vec2)):
        trial_circuit.u1(-2*x_vec2[r], q[r])
        trial_circuit.h(q[r])

    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])

    return trial_circuit


# def inner_prod_circuit_ML(entangler_map, coupling_map, initial_layout, n, x_vec1, x_vec2,
#                           meas_string=None, measurement=True):
#
#     q = QuantumRegister(n, "q")
#     c = ClassicalRegister(n, "c")
#     trial_circuit = QuantumCircuit(q, c)
#
#     # write input state from sample distribution
#     for r in range(len(x_vec1)):
#         trial_circuit.u2(0.0, np.pi, q[r])  # h
#         trial_circuit.u1(2*x_vec1[r], q[r])
#     for node in entangler_map:
#         for j in entangler_map[node]:
#             trial_circuit.cx(q[node], q[j])
#             trial_circuit.u1(2*(np.pi-x_vec1[node])*(np.pi-x_vec1[j]), q[j])
#             trial_circuit.cx(q[node], q[j])
#
#     for r in range(len(x_vec1)):
#         trial_circuit.u2(0.0, np.pi, q[r])  # h
#         trial_circuit.u1(2*x_vec1[r], q[r])
#     for node in entangler_map:
#         for j in entangler_map[node]:
#             trial_circuit.cx(q[node], q[j])
#             trial_circuit.u1(2*(np.pi-x_vec1[node])*(np.pi-x_vec1[j]), q[j])
#
#     for node in entangler_map:
#         for j in entangler_map[node]:
#             trial_circuit.u1(-2*(np.pi-x_vec2[node])*(np.pi-x_vec2[j]), q[j])
#             trial_circuit.cx(q[node], q[j])
#     for r in range(len(x_vec2)):
#         trial_circuit.u1(-2*x_vec2[r], q[r])
#         trial_circuit.u2(0.0, np.pi, q[r])  # h
#
#     for node in entangler_map:
#         for j in entangler_map[node]:
#             trial_circuit.cx(q[node], q[j])
#             trial_circuit.u1(-2*(np.pi-x_vec2[node])*(np.pi-x_vec2[j]), q[j])
#             trial_circuit.cx(q[node], q[j])
#     for r in range(len(x_vec2)):
#         trial_circuit.u1(-2*x_vec2[r], q[r])
#         trial_circuit.u2(0.0, np.pi, q[r])  # h
#
#     if measurement:
#         for j in range(n):
#             trial_circuit.measure(q[j], c[j])
#
#     return trial_circuit


my_zero_string = ''


def get_zero_string(num_of_qubits):
    global my_zero_string

    if len(my_zero_string) != 0:
        return my_zero_string

    for nq in range(num_of_qubits):
        my_zero_string += '0'
    return my_zero_string


def kernel_join(points_array, points_array2, entangler_map, coupling_map, initial_layout,
                shots, seed, num_of_qubits, backend):
    Q_program = QuantumProgram()
    circuits = []
    is_identical = np.all(points_array == points_array2)

    my_zero_string = get_zero_string(num_of_qubits)
    if is_identical:  # we reduce the computation by leveraging the symmetry of matrix: compute only the upper-right corner
        size = len(points_array)
        my_product_list = list(itertools.combinations(range(len(points_array)), 2))
        for a in range(len(my_product_list)):
            first_index = my_product_list[a][0]   # This number is first datapoint in product
            second_index = my_product_list[a][1]  # This number is second datapoint in product
            sequencesp = inner_prod_circuit_ML(entangler_map, coupling_map, initial_layout, num_of_qubits,
                                               points_array[first_index], points_array[second_index], None, True)

            circuit_name = 'join_'+str(first_index) + "_" + str(second_index)
            circuits.append(circuit_name)
            Q_program.add_circuit(circuit_name, sequencesp)

        circuit_list = [c for c in circuits]
        program_data = Q_program.execute(circuit_list, backend=backend, coupling_map=coupling_map,
                                         initial_layout=initial_layout, shots=shots, seed=seed, timeout=600)

        mat = np.eye(size, size)  # element on the diagonal is always 1: point*point=|point|^2
        for v in range(len(program_data)):
            circuit_name = circuits[v]
            tmp = circuit_name.split('_')
            first_index = int(tmp[1])
            second_index = int(tmp[2])
            countsloop = program_data.get_counts(circuit_list[v])

            if my_zero_string in countsloop:
                mat[first_index][second_index] = countsloop[my_zero_string]/shots
            else:
                mat[first_index][second_index] = 0
            mat[second_index][first_index] = mat[first_index][second_index]
        return mat
    else:
        Q_program = QuantumProgram()
        total_test = points_array
        svm = points_array2

        for a in range(len(total_test)):
            for b in range(len(svm)):
                sequencesp = inner_prod_circuit_ML(entangler_map, coupling_map, initial_layout, num_of_qubits,
                                                   svm[b], total_test[a], None, True)
                cp = 'join_' + str(a) + "_" + str(b)
                circuits.append(cp)
                Q_program.add_circuit(cp, sequencesp)

        circuit_list = [c for c in circuits]
        program_data = Q_program.execute(circuit_list, backend=backend, coupling_map=coupling_map,
                                            initial_layout=initial_layout, shots=shots, seed=seed)

        mat = np.zeros((len(total_test), len(svm)))
        for v in range(len(program_data)):
            countsloop = program_data.get_counts(circuit_list[v])
            if my_zero_string in countsloop:
                circuit_name = circuits[v]
                tmp = circuit_name.split('_')
                first_index = int(tmp[1])
                second_index = int(tmp[2])
                countsloop = program_data.get_counts(circuit_list[v])

                if my_zero_string in countsloop:
                    mat[first_index][second_index] = countsloop[my_zero_string]/shots
                else:
                    mat[first_index][second_index] = 0

        return mat

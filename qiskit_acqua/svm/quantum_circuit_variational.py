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

from qiskit.tools.apps.optimization import trial_circuit_ryrz,trial_circuit_ry
from qiskit import QuantumCircuit, QuantumProgram, QuantumRegister, ClassicalRegister, extensions
from cost_helpers import *
import numpy as np
import operator


def trial_circuit_ML(entangler_map, coupling_map, initial_layout,n, m, theta, x_vec, name = 'circuit',\
                     meas_string = None, measurement = True):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m*2 stacked as theta[n*i*2+2*j+p] where j
    counts the qubits and i the depth and p if y and z.
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    pauli_string = length of number of qubits string
    """
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
    trial_circuit = QuantumCircuit(q, c)
    
    #write input state from sample distribution
    for feat_map in range(2):
        for r in range(len(x_vec)):
            trial_circuit.h(q[r])
            trial_circuit.u1(2*x_vec[r], q[r])
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cx(q[node], q[j])
                trial_circuit.u1(2*(np.pi-x_vec[node])*(np.pi-x_vec[j]), q[j])
                trial_circuit.cx(q[node], q[j])
        
    trial_circuit.barrier(q)
    for r in range(len(x_vec)):
#         trial_circuit.u3(theta[r * 3], theta[r * 3 + 1], theta[r * 3+ 2], q[r])
        trial_circuit.ry(theta[2*r], q[r])
        trial_circuit.rz(theta[2*r + 1], q[r])
    for i in range(m):
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
#             trial_circuit.u3(theta[n * (i+1) * 3 + 3*j], theta[n * (i+1) * 3 + 3*j + 1], theta[n * (i+1) * 3 + 3*j + 2], q[j])
            trial_circuit.ry(theta[n * (i+1) * 2 + 2*j], q[j])
            trial_circuit.rz(theta[n * (i+1) * 2 + 2*j + 1], q[j])
    trial_circuit.barrier(q)

    if measurement:
        for j in range(n):
#             trial_circuit.h(q[j])
            trial_circuit.measure(q[j], c[j])
    return name, trial_circuit




def eval_cost_function(entangler_map, coupling_map, initial_layout,n,m,x_vec,class_labels, \
                       backend,shots,train,theta):
    sample_shots = 0
    #x_vec is the vector of training characteristics - size n
    #y is the binary outcome for each x_vec
    number_of_classes = len(class_labels)
    cost=0
    total_cost = 0
    std_cost = 0

    predicted_results = []

    Q_program = QuantumProgram()
#     Q_program.set_api(Qconfig.APItoken,Qconfig.config["url"])

    ### RUN CIRCUITS

    circuits = []
    c = []
    sequences = []

    unlabeled_data = []
    for arrays in range(number_of_classes):
        labelgroup = x_vec[class_labels[arrays]]
        for item in labelgroup:
            unlabeled_data.append(item)

    # unlabeled_data = np.array([])
    # for arrays in range(number_of_classes):
    #     unlabeled_data = np.vstack((unlabeled_data,x_vec[class_labels[arrays]])) if unlabeled_data.size else x_vec[class_labels[arrays]]



    for key in x_vec.keys():
        for c_id, inpt in enumerate(x_vec[key]):
            c, sequences = trial_circuit_ML(entangler_map, coupling_map, initial_layout,n,m,\
                                            theta,inpt,key+str(c_id),None,True)
            #c is 'A0', 'A1', 'A2', etc...
            circuits.append((key, c))
            Q_program.add_circuit(c,sequences)

    circuit_list  = [c[1] for c in circuits]

#     # circuits is ['A', 'A0'], ['A', 'A1'], ..., ['B', 'B0'],... etc
#     print(Q_program.get_qasm(circuit_list[0]))

    program_data = Q_program.execute(circuit_list,backend=backend, coupling_map=coupling_map, initial_layout=initial_layout, shots=shots)
    result = 0
    success = 0
    for v in range(len(circuit_list)):
        countsloop = program_data.get_counts(circuit_list[v])

        for cl in range(number_of_classes):
            if circuits[v][0] == class_labels[cl]:
                expected_outcome = class_labels[cl]

        probs = return_probabilities(countsloop, class_labels)
        result += cost_estimate_sigmoid(200, probs, expected_outcome)

        predicted_results.append(max(probs.items(), key=operator.itemgetter(1))[0])
        if max(probs.items(), key=operator.itemgetter(1))[0] == expected_outcome:
            success += 1

        if not train:
            print("\n=============================================\n")
            print('Classifying point %s. Label should be  %s \n'%(unlabeled_data[v] ,expected_outcome))
            print('Measured label probability distribution is %s \n'%probs)
            if max(probs.items(), key=operator.itemgetter(1))[0] == expected_outcome:
                print ('Assigned label is  %s  CORRECT \n'%max(probs.items(), key=operator.itemgetter(1))[0])
            else:
                print ('Assigned label is  %s INCORRECT \n'%max(probs.items(), key=operator.itemgetter(1))[0])

    total_cost = result/len(circuit_list)
    success_ratio = success/len(circuit_list)

    return total_cost, std_cost, success_ratio, predicted_results




def eval_cost_function_with_unlabeled_data(entangler_map, coupling_map, initial_layout,n,m,unlabeled_data,class_labels, \
                       backend,shots,train,theta):
    predicted_results = []
    Q_program = QuantumProgram()
    circuits = []

    for c_id, inpt in enumerate(unlabeled_data):
        circuit_name, sequences = trial_circuit_ML(entangler_map, coupling_map, initial_layout,n,m,\
                                        theta,inpt,''+str(c_id),None,True)
        circuits.append(('', circuit_name))
        Q_program.add_circuit(circuit_name,sequences)

    circuit_list  = [c[1] for c in circuits]
    program_data = Q_program.execute(circuit_list,backend=backend, coupling_map=coupling_map, initial_layout=initial_layout, shots=shots)

    for v in range(len(circuit_list)):
        countsloop = program_data.get_counts(circuit_list[v])
        probs = return_probabilities(countsloop, class_labels)
        predicted_results.append(max(probs.items(), key=operator.itemgetter(1))[0])


    return predicted_results

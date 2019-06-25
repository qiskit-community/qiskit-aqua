#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:40:05 2018

@author: ssheldo
"""

from scipy.optimize import minimize
import scipy.linalg as la
import numpy as np
# import sys
import qiskit.tools.qcvv.tomography as tomo
from qiskit import QuantumCircuit
# sys.path.append("../../qiskit-sdky-py/")

def generate_A_matrix(results, circuits, qubits, shots):
    # documentation
    qubits = sorted(qubits)
    cals = dict()
    for ii in range(2**len(qubits)):
        cals['%s' % ii] = tomo.marginal_counts(results[circuits[ii].name], qubits)
    A = []
    for ii in range(2**len(qubits)):
        A.append([cals['%s' % ii][state]/shots for state in cals['%s' % ii]])
    A = np.transpose(np.array(A))
    #measurement = la.pinv(A)
    return A

def generate_A_matrix_bk(results, circuits, qubits, shots):
    # documentation
    qubits = sorted(qubits)
    cals = dict()
    for ii in range(2**len(qubits)):
        cals['%s' % ii] = tomo.marginal_counts(results.get_counts(circuits[ii]), qubits)
    A = []
    for ii in range(2**len(qubits)):
        A.append([cals['%s' % ii][state]/shots for state in cals['%s' % ii]])
    A = np.transpose(np.array(A))
    #measurement = la.pinv(A)
    return A


def remove_measurement_errors(results, circuit, qubits, A, shots, method=1, data_format='counts'):
    """

        results (dict)
        circuit (QuantumCircuit)
        qubits (list)
        A (np.ndarray)
        shots (number)
        data_format (str):
    """

    if A is None:
        return results[circuit.name]

    data = tomo.marginal_counts(results[circuit.name], qubits)
    datavec = np.array([data[state]/shots for state in data])
    states = list(data.keys())
    if method == 0:
        data_processed_vec = np.dot(la.pinv(A), datavec)

    if method == 1:
        def fun(x): return sum((datavec - np.dot(A, x))**2)
        x0 = np.random.rand(len(datavec))
        cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
        bnds = tuple((0, 1) for x in x0)
        res = minimize(fun, x0, method='SLSQP', constraints=cons, bounds=bnds, tol=1e-6)
        data_processed_vec = shots*res.x
        data_processed_vec = shots*res.x
    if data_format is 'vec':
        data_processed = data_processed_vec
    elif data_format is 'counts':
        data_processed = {states[i]: data_processed_vec[i] for i in range(len(states))}
    return data_processed


def insert_cals_allstates(qp, circuits, qubits, q, c):
    for j in range(2**len(qubits)):
        circuit = qp.create_circuit("circuit_%s" % j, [q], [c])
        binnum = np.binary_repr(j)
        for k in range(len(binnum)):
            if binnum[len(binnum)-1-k] == '1':
                circuit.x(q[qubits[k]])
        for k in range(len(qubits)):
            circuit.measure(q[qubits[k]], c[qubits[k]])
        circuits.insert(j, "circuit_%s" % j)


def make_cal_circuits(qubits, qr, cr, cbits=0):
    """
        qubits: list of qubit index
        qr (QuantumRegister):
        cr: (ClassicalRegister):

        [QuantumCircuit]
    """
    qubit_mapper = {0: 2, 1: 1, 2: 0, 3: 5}
    cals = []
    if cbits == 0:
        cbits = qubits
    for j in range(2**len(qubits)):
        circuit = QuantumCircuit(qr, cr)
        binnum = np.binary_repr(j)
        for k in range(len(binnum)):
            if binnum[len(binnum)-1-k] == '1':
                circuit.x(qr[qubit_mapper[qubits[k]]])
        for k in range(len(qubits)):
            circuit.measure(qr[qubit_mapper[qubits[k]]], cr[cbits[k]])
        cals.append(circuit)
    return cals


class TempResult:

    def __init__(self):
        self.circuit_to_counts = {}

    def add_result(self, circuit, counts):
        self.circuit_to_counts[circuit.name] = counts

    def get_counts(self, circuit):
        return self.circuit_to_counts[circuit.name]

def remove_measurement_errors_all(results, circuits, qubits, A, shots, method=1, data_format='counts'):
    temp_result = TempResult()
    for circuit in circuits:
        counts = remove_measurement_errors(results, circuit, qubits, A, shots, method=method, data_format=data_format)
        temp_result.add_result(circuit, counts)

    return temp_result


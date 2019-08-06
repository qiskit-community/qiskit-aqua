from qiskit import Aer, execute, QuantumCircuit
from qiskit.aqua.algorithms import ExactEigensolver
import numpy as np
from qiskit.aqua.operator import Operator

def to_unitary(self):
    job = execute(self, Aer.get_backend('unitary_simulator'))
    return job.result().get_unitary()

def to_statevector(self):
    job = execute(self, Aer.get_backend('statevector_simulator'))
    return job.result().get_statevector()

def to_probabilities(self):
    return np.abs(self.to_statevector())**2

QuantumCircuit.to_unitary = to_unitary
QuantumCircuit.to_statevector = to_statevector
QuantumCircuit.to_probabilities = to_probabilities

def diagonalize(self, k):
    exact_eigensolver = ExactEigensolver(self, k=k)
    return(exact_eigensolver.run())

Operator.diagonalize = diagonalize
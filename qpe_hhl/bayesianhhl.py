import numpy as np
from math import log
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, register, execute#, ccx
from qiskit import register, available_backends, get_backend, least_busy
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance

from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer, plot_histogram
import matplotlib.pyplot as plt

backend = 'local_qasm_simulator'
n = 5
inputregister = QuantumRegister(n, name="inputregister")
flagbit = QuantumRegister(1, name="flagbit")
outputregister = QuantumRegister(n, name="outputregister")
anc = QuantumRegister(n-1, name="anc")
classreg1 = ClassicalRegister(n)
classreg2 = ClassicalRegister(1)
classreg3 = ClassicalRegister(n)

qc = QuantumCircuit(inputregister, flagbit,  anc, outputregister, classreg1, classreg2, classreg3)

qc.x(inputregister[4])
qc.x(inputregister[1])
qc.cx(inputregister[0], flagbit)
qc.cx(inputregister[0], outputregister[n-1])

for i in range(1,n):
    qc.x(flagbit[0])
    qc.ccx(inputregister[i], flagbit[0], outputregister[n-1-i])
    qc.x(flagbit)
    qc.cx(outputregister[n-1-i], flagbit)

#several control registers for the not gate of the flagbit at the end
qc.ccx(outputregister[0], outputregister[1], anc[0])
for i in range(2, n):
    qc.ccx(outputregister[i], anc[i-2], anc[i-1])

# copy
qc.cx(anc[n-2], flagbit)

# uncompute
for i in range(n-1, 1, -1):
    qc.ccx(outputregister[i], anc[i-2], anc[i-1])
qc.ccx(outputregister[0], outputregister[1], anc[0])

qc.x(flagbit)
qc.barrier(inputregister, flagbit, outputregister)
qc.measure(inputregister, classreg1)
qc.measure(flagbit, classreg2)
qc.measure(outputregister, classreg3)

#results = execute(qc, backend = backend).result()
#print(results._result)
#print(results.get_counts(qc))
drawer(qc, filename="firstnewton.png")
#plt.show()
#plot_histogram(results.get_counts())



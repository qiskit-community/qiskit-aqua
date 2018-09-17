from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt
inputreg = QuantumRegister(8)
outputreg = QuantumRegister(8)
meas = ClassicalRegister(16)

qc = QuantumCircuit(inputreg, outputreg, meas)

qc.x(outputreg[0])

qc.cx(outputreg[0], outputreg[1])

qc.x(inputreg[7])
qc.ccx(inputreg[6], inputreg[7], outputreg[0])
qc.x(inputreg[7])

qc.cx(outputreg[0], outputreg[1])

qc.x(inputreg[7])
qc.x(inputreg[6])
qc.ccx(inputreg[6], inputreg[7], outputreg[0])
qc.x(inputreg[7])
qc.x(inputreg[6])

qc.cx(outputreg[0], outputreg[3])

qc.cx(outputreg[0], outputreg[2])

drawer(qc)
plt.show()





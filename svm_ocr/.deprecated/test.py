import numpy as np
from time import sleep  # used for polling jobs
    
# importing the QISKit
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, QuantumProgram

# import tomography library
import qiskit.tools.qcvv.tomography as tomo

# Create a 2-qubit quantum register
qr = QuantumRegister(2)
cr = ClassicalRegister(2)

x = [(), ()]

x[0] = (0.987, 0.159)
x[1] = (0.354, 0.935)

t1 = 2*np.arctan(x[0][1]/x[0][0])
t2 = 2*np.arctan(x[1][1]/x[1][0])

# quantum circuit to make an entangled Bell state 
bell = QuantumCircuit(qr, cr, name='bell')
bell.h(qr[0])
bell.x(qr[0])
bell.cu3(t1, 0, 0, qr[0], qr[1])
bell.x(qr[0])
bell.cu3(t2, 0, 0, qr[0], qr[1])

# Construct state tomography set for measurement of qubits [0, 1] in the Pauli basis
bell_tomo_set = tomo.state_tomography_set([0])

# Create a quantum program containing the state preparation circuit
Q_program = QuantumProgram()
Q_program.add_circuit('bell', bell)

# Add the state tomography measurement circuits to the Quantum Program
bell_tomo_circuit_names = tomo.create_tomography_circuits(Q_program, 'bell', qr, cr, bell_tomo_set)

print('Created State tomography circuits:')
for name in bell_tomo_circuit_names:
    print(name)


# Use the local simulator
backend = 'local_qasm_simulator'

# Take 5000 shots for each measurement basis
shots = 5000

# Run the simulation
bell_tomo_result = Q_program.execute(bell_tomo_circuit_names, backend=backend, shots=shots)
print(bell_tomo_result)

bell_tomo_data = tomo.tomography_data(bell_tomo_result, 'bell', bell_tomo_set)

rho_fit = tomo.fit_tomography_data(bell_tomo_data)
print(rho_fit)

r = np.array([[np.dot(x[i], x[j]) for i in range(2)] for j in range(2)])
print(r/np.trace(r))

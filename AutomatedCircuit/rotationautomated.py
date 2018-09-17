from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
import parserevkit as parse

n = 8

qr1 = QuantumRegister(n)
qr2 = QuantumRegister(n)
cr1 = ClassicalRegister(n)
cr2 = ClassicalRegister(n)
qc = QuantumCircuit(qr1, qr2, cr1, cr2)
qc.x(qr1[3])
qc.x(qr1[1])
#qc.x(qr1[0])
qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)
parse.measure(qc,n,qr1, qr2, cr1, cr2)

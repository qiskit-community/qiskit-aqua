from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
import parserevkit as parse

n = 5

qr1 = QuantumRegister(n)
qr2 = QuantumRegister(n)
ancilla = QuantumRegister(1)
cr1 = ClassicalRegister(n)
cr2 = ClassicalRegister(n)
cr3 = ClassicalRegister(1)
qc = QuantumCircuit(qr1, qr2, cr1, cr2, ancilla, cr3)
qc.x(qr1[3])
qc.x(qr1[1])
#qc.x(qr1[0])
qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)
#parse.measure(qc,n,qr1, qr2, cr1, cr2)

c = 1./2**n
for i in range(n):
    qc.cu3(c*0.5, 0, 0, qr1[i], ancilla)

qc.measure(qr1, cr1)
qc.measure(qr2, cr2)
qc.measure(ancilla, cr3)
job = execute(qc, backend='local_qasm_simulator', shots = 8000)
print(job.result().get_counts())


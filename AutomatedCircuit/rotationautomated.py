from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
import parserevkit as parse
import math

n = 6
shots = 1000
z = 20
mean = 0
for j in range(z):
	print(j)
	qr1 = QuantumRegister(n)
	qr2 = QuantumRegister(n)
	ancilla = QuantumRegister(1)
	cr1 = ClassicalRegister(n)
	cr2 = ClassicalRegister(n)
	cr3 = ClassicalRegister(1)
	qc = QuantumCircuit(qr1, qr2, cr1, cr2, ancilla, cr3)
	#qc.x(qr1[3])
	qc.x(qr1[1])
	#qc.x(qr1[0])
	qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)
	#parse.measure(qc,n,qr1, qr2, cr1, cr2)

	c = 10.
	for i in range(1, n+1):
		qc.cu3(2*c*0.5*2**(-i), 0, 0, qr2[n-i], ancilla)

	#qc.h(ancilla)
	qc.measure(qr1, cr1)
	qc.measure(qr2, cr2)
	qc.measure(ancilla, cr3)
	job = execute(qc, backend='local_qasm_simulator', shots = shots)
	print(job.result().get_counts())
	rd = job.result().get_counts()
	element = [k for k in rd]
	probs = [rd[k] for k in rd]

	real = sum([2**(i) for i, e in enumerate(reversed(element[1])) if e == "1" and i < n])
	inverse = sum([2**-(i-1) for i, e in enumerate(element[1]) if e == "1" and i > 1 and i <= n+1])

	prob = probs[1]/shots
	realprob = 0.5*c/real
	mean += math.sqrt(prob)
	print("Input number = ", real)
	print("Reciprocal number = ", inverse)
	print("True C/lambda = ", realprob)
	print("Measured C/lambda = ", math.sqrt(prob))

print("Mean = ", mean/z)

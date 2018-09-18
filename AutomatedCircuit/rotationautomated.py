from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
import parserevkit as parse
import numpy as np
from qpe import QPE
import math

n = 5
shots = 10000
z = 20
mean = 0
qpe = QPE()
hermitian_matrix = True
backend = 'local_qasm_simulator'
matrix = [[3, -1],[-1,3]]
matrix = np.array(matrix)
invec = [-1, 1]
params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': 5,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'hermitian_matrix': hermitian_matrix,
        'backend' : backend
},
"iqft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": invec
}}

qpe.init_params(params, matrix)


#qc = qpe._compute_eigenvalue()
#res = qpe._ret
qc, qr1 = qpe._setup_qpe()
#rets = qpe._compute_eigenvalue

#qr1 = QuantumRegister(n)
qr2 = QuantumRegister(n)
ancilla = QuantumRegister(1)
cr1 = ClassicalRegister(n)
cr2 = ClassicalRegister(n)
cr3 = ClassicalRegister(1)
qc2 = QuantumCircuit(qr2, cr1, cr2, ancilla, cr3)

qc += qc2
#qc.x(qr1[3])
#qc.x(qr1[1])	
#qc.x(qr1[0])
qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)
#parse.measure(qc,n,qr1, qr2, cr1, cr2)

c = 2.
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

real = sum([2**(i) for i, e in enumerate(reversed(element[1])) if e == "1" and i < n])*1/2**n*(2*np.pi/qpe._evo_time)
inverse = sum([2**-(i-1) for i, e in enumerate(element[1]) if e == "1" and i > 1 and i <= n+1])*2**n/(2*np.pi/qpe._evo_time)

prob = probs[1]/shots
realprob = 0.5*c/real
#mean += math.sqrt(prob)
print("Input number = ", real)
print("Reciprocal number = ", inverse)
print("True C/lambda = ", realprob)
print("Measured C/lambda = ", math.sqrt(prob))


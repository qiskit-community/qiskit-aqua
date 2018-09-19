from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
import parserevkit as parse
import numpy as np
from qpe import QPE
import math

n = 7
shots = 10000
z = 10
mean = {}
realdict = {}
inversedict = {}
realprob = {}
qpe = QPE()
hermitian_matrix = True
backend = 'local_qasm_simulator'
matrix = [[1, 0, 0, 0],[0,2,0,0],[0,0,3,0],[0,0,0,3]]
#matrix = [[3, -1],[-1,3]]
matrix = np.array(matrix)
invec = [0,1,0,0]
params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': n,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'hermitian_matrix': hermitian_matrix,
        'backend' : backend,
        'evo_time' : (1-2**-n)*2*np.pi/3.
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
evo_time = (1-2**-n)*2*np.pi/3.
#print(res)
#qr1 = QuantumRegister(n)
qr2 = QuantumRegister(n)
ancilla = QuantumRegister(1)
cr1 = ClassicalRegister(n)
cr2 = ClassicalRegister(n)
cr3 = ClassicalRegister(1)
qc2 = QuantumCircuit(qr2, cr1, cr2, ancilla, cr3)

qc += qc2
#qc.x(qr1[4])
#qc.x(qr1[2])	
#qc.x(qr1[0])
qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)
#parse.measure(qc,n,qr1, qr2, cr1, cr2)

c = 1.
for i in range(1, n+1):
	qc.cu3(2*c*0.5*2**(-i), 0, 0, qr2[n-i], ancilla)

#qc.h(ancilla)
#qc += qpe._construct_inverse()
qc.measure(qr1, cr1)
qc.measure(qr2, cr2)
qc.measure(ancilla, cr3)
for l in range(z):
    print(l)
    job = execute(qc, backend='local_qasm_simulator', shots = shots)
    print(job.result().get_counts())
    rd = job.result().get_counts()
    element = [k for k in rd]
    probs = [rd[k] for k in rd]
    for k in element:
        if k not in realprob:
            realprob.update({str(k): 0})
        if k not in mean:
            mean.update({str(k): 0})
            inversedict.update({str(k):0})
            realdict.update({str(k): 0})
    #print(len(element))
    a = 0
    for i in range(len(element)):
        #print(element[i][0])
        if element[i][0] == "1": #and sum([int(e) for i, e in enumerate(reversed(element[i])) if i < n]) == 0:
            real = sum([2**(i) for i, e in enumerate(reversed(element[i])) if e == "1" and i < n])*(2*np.pi/evo_time)/2**n
            inverse = sum([2**-(i-1) for i, e in enumerate(element[i]) if e == "1" and i > 1 and i <= n+1])*(2**n)/(2*np.pi/evo_time)
            print(element[i])
            print("real = ",real)
            print("inverse = ", inverse)
            
            prob = math.sqrt(probs[i]/shots*(2**n/(2*np.pi/evo_time))**2)
            print(prob)
            print(0.5*c*inverse)
            new = realprob[str(element[i])] + 0.5*c*inverse
            new2 = mean[str(element[i])] + prob
            realprob.update({str(element[i]) : new})
            mean.update({str(element[i]) : new2})
            inversedict.update({str(element[i]) : inverse})
            realdict.update({str(element[i]) : real})
            #print("Reciprocal number = ", inverse)
            #print("True C/lambda = ", realprob)
            #print("Measured C/lambda = ", math.sqrt(prob))
            
for k in mean:
    if mean[k] != 0:
        print("real ",k, realdict[k])
        print("inverse ", k, inversedict[k])
        print("Calculated prob ", k, mean[k])
for k in realprob:
    if realprob[k] != 0:
        print("Real prob ", k, realprob[k])


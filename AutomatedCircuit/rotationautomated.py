from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
from qiskit.extensions.simulator import snapshot
import matplotlib.pyplot as plt
import parserevkit as parse
import numpy as np
from qpe import QPE
import math
import sys
from qiskit_aqua.utils import random_matrix_generator as rmg
#sys.path.append('afs/bb/u/isabel-h/drive/qiskit-aqua-hhl/matrix-gen/')
#import afs.bb/u/isabel-h/drive/qiskit-aqua-hhl/matrix-gen/matrix_gen

def testing(dicta, dictb, n, evo_time):
    statevec = [k for k in dicta[0]]
    element = [k for k in dictb[0]]
    real = []
    realprob = []
    inverse = []
    prob = []
    calc = []
    for i in range(len(statevec)):
        real.append(sum([2**(n-1-a) for a, e in enumerate(reversed(statevec[i])) if e == "1" and a < n])*(2*np.pi/evo_time)/2**n)
        realprob.append(np.sqrt(dicta[0][statevec[i]][0]**2+dicta[0][statevec[i]][1]**2))
    for i in range(len(element)):
        if element[i][0] == "1":
            inverse2 = sum([2**(-(a-1)) for a, e in enumerate(element[i]) if e == "1" and a > 1 and a <= n+1])*(2**n)/(2*np.pi)*evo_time
            inverse.append(inverse2)
            prob.append(np.sqrt(dictb[0][element[i]][0]**2+dictb[0][element[i]][1]**2))
    inverse = np.array(inverse)
    prob = np.absolute(np.array(prob))
    return(inverse, prob, real, realprob)

n = 5
shots = 10000
z = 1
#mean = {}
#realdict = {}
#inversedict = {}
#realprob = {}
qpe = QPE()
hermitian_matrix = True
backend = 'local_qasm_simulator'
matrix = [[1, 0, 0, 0],[0,2,0,0],[0,0,3,0],[0,0,0,3]]
#matrix = [[3, -1],[-1,3]]
matrix = rmg.random_hermitian(4, eigrange = [1, 4], sparsity = 0.5)
matrix = np.array(matrix)
w,v = np.linalg.eig(matrix)
print(w[1])
print("inverse = ", 1./ w[1])
invec = v[1]
print(v[1])
params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': n,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'hermitian_matrix': hermitian_matrix,
        'backend' : backend
        #'evo_time' : (1-2**-n)*2*np.pi/10.
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
qc.snapshot("1")
#print(res)
#qr1 = QuantumRegister(n)
evo_time = qpe._evo_time
print(evo_time)
qr2 = QuantumRegister(n)
ancilla = QuantumRegister(1)
cr1 = ClassicalRegister(n)
cr2 = ClassicalRegister(n)
cr3 = ClassicalRegister(1)
qc2 = QuantumCircuit(qr2, cr1, cr2, ancilla, cr3)

qc += qc2

#for i in range(int(n/2)):
#    qc.swap(qr1[i], qr1[n-1-i])
qc, qr1, qr2, cr1, cr2 = parse.read_make_circuit(n, qr1, qr2, cr1, cr2, qc)

c = 2.
for i in range(1, n+1):
	qc.cu3(2*c*0.5*2**(-i), 0, 0, qr2[n-i], ancilla)

qc += qpe._construct_inverse()
#qc.measure(qr1, cr1)
#qc.measure(qr2, cr2)
#qc.measure(ancilla, cr3)
qc.snapshot("2")

res = execute(qc,backend="local_qasm_simulator",config={"data":["quantum_state_ket"]},shots=1)
res1 = res.result().get_data()["snapshots"]["1"]["quantum_state_ket"]

res_ = res.result().get_data()["snapshots"]["2"]["quantum_state_ket"]
inverse, prob, real, realprob = testing(res1, reas_, n, evo_time)
idx = np.argmax(prob)
print("inverse = ", inverse[idx], " Prop = ", prob[idx], " Calc = ", calc[idx])
#plt.plot(inverse, calc, "go")
plt.subplot(121)
plt.plot(inverse, prob, "ro")
plt.title("Inverse")
plt.subplot(122)
plt.plot(real, realprob, "go")
plt.title("Real")
plt.show()
plt.savefig("rottest.png") 
"""for l in range(z):
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
        if element[i][0] == "1" and sum([int(e) for i, e in enumerate(reversed(element[i])) if i < n]) == 0:
            #real = sum([2**(n-1-a) for a, e in enumerate(reversed(element[i])) if e == "1" and a < n])*(2*np.pi/evo_time)/2**n
            inverse = sum([2**(-(a-1)) for a, e in enumerate(element[i]) if e == "1" and a > 1 and a <= n+1])*(2**n)/(2*np.pi)*evo_time
            #print([2**(-(a-1)) for a, e in enumerate(element[i]) if e == "1" and a > 1 and a <= n+1])
            print(element[i])
            #print("real = ",real)
            print("inverse = ", inverse)
            
            prob = math.sqrt(probs[i]/shots*(2**n/(2*np.pi)*evo_time)**2)
            print("Meas prob = ", prob)
            print("Real prob = ", 0.5*c*inverse)
            new = realprob[str(element[i])] + 0.5*c*inverse
            new2 = mean[str(element[i])] + prob
            realprob.update({str(element[i]) : new})
            mean.update({str(element[i]) : new2})
            inversedict.update({str(element[i]) : inverse})
            #realdict.update({str(element[i]) : real})
            #print("Reciprocal number = ", inverse)
            #print("True C/lambda = ", realprob)
            #print("Measured C/lambda = ", math.sqrt(prob))
            
for k in mean:
    if mean[k] != 0:
        #print("real ",k, realdict[k])
        print("inverse ", k, inversedict[k])
        print("Measured prob ", k, mean[k])
        print("Real prob ", k, realprob[k])
        print("---------------------------------------------")"""


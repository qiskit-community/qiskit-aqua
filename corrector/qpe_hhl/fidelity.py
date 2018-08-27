from qpe import QPE
from qiskit_aqua import Operator
from qiskit import register, execute
from qiskit.extensions.simulator import snapshot
import numpy as np
from qiskit import available_backends
import sys
sys.path.append("..")
from matrix_gen import gen_matrix

import matplotlib.pyplot as plt

qpe = QPE()
n = 2
k = 3
nege = False

matrix = gen_matrix(n, eigrange=[0, 5], trunc=3)
#matrix = np.diag([1, 2])
#np.save("mat.npy", matrix)
#matrix = np.load("mat.npy")
w, v = np.linalg.eigh(matrix) 

print(matrix)
print("Eigenvalues:", w)
print("Eigenvectors:", v)

beta = np.array([1, 1])

beta = beta/np.sqrt(beta.dot(beta.conj()))

invec = v.dot(beta)

params = {
    'algorithm': {
            'name': 'QPE',
            'num_ancillae': k,
            'num_time_slices': 20,
            'expansion_mode': 'suzuki',
            'expansion_order': 3,
            'hermitian_matrix': True,
            'negative_evals': nege,
            #'evo_time': 2*np.pi/2,
            #'use_basis_gates': False,
    },
    "iqft": {
        "name": "STANDARD"
    },
    "initial_state": {
        "name": "CUSTOM",
        "state_vector": invec#[1/2**0.5,1/2**0.5]
    }
}

def create_theory_state_vector(k, n, w, v, beta, t):
    coef = np.zeros((2**k*n, n), dtype=np.complex128)
    print(t)
    for j in range(n):
        coefs = np.zeros(2**k, dtype=np.complex128)
        for y in range(2**k):
            coefs[y] = 1/2**k*sum([np.exp(1j*(w[j]*t-2*np.pi*y/2**k)*x) for x in range(2**k)])
            if np.isnan(coefs[y]):
                coefs[y] = 1
                print(j ,beta[j], coefs[y])
        coef[:, j] = np.kron(v[j], coefs)
        print(j, coef[:, j].dot(coef[:, j].conj()))
    return coef.dot(beta)


qpe.init_params(params, matrix)

qc = qpe._setup_qpe(measure=False)
a = qc.get_qregs()["a"]
for i in range(int(k/2)):
    qc.swap(a[i], a[k-i-1])

backend = "local_qasm_simulator"
if backend == "local_qasm_simulator":
    qc.snapshot("1")
    res = execute(qc, "local_qasm_simulator", config={"data": [], "shots": 1})
    sv = res.result().get_data()["snapshots"]["1"]["statevector"][0]
elif backend == "local_statevector_simulator":
    res = execute(qc, "local_statevector_simulator")
    sv = res.result().get_data()["statevector"]
#print("Statevector\n", sv.round(3))

tv = create_theory_state_vector(k, n, w, v, beta, qpe._evo_time)
#print("Theoryvector\n", tv.round(3))

fidelity = abs(sv.dot(tv.conj()))**2

probs = abs(sv)
probt = abs(tv)

print("Fidlity:", fidelity)
print("Prob fidelity:", probt.dot(probs)) 

print(sv.dot(sv.conj()), tv.dot(tv.conj()), beta.dot(beta.conj()))

import matplotlib.pyplot as plt

#plt.plot(range(len(sv)), sv.real)
#plt.plot(range(len(sv)), sv.imag)
#plt.plot(range(len(tv)), tv.real)
#plt.plot(range(len(tv)), tv.imag)
#plt.plot(range(len(sv)), probs)
#plt.plot(range(len(tv)), probt)
#plt.show()


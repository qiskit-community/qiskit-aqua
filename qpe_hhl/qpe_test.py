from qpe import QPE
from qiskit_aqua import Operator
import scipy
from qiskit import register
import numpy as np
from qiskit import available_backends
import sys
sys.path.append("..")
from matrix_gen import gen_matrix

import matplotlib.pyplot as plt

qpe = QPE()
n = 2
k = 9
nege = True

matrix = gen_matrix(n, eigrange=[-5, 5], sparsity=0.6)
#matrix = np.diag([-1.5, 1])
#np.save("mat.npy", matrix)
#matrix = np.load("mat.npy")
w, v = np.linalg.eigh(matrix) 

print(matrix)
print("Eigenvalues:", w)

invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))

params = {
    'algorithm': {
            'name': 'QPE',
            'num_ancillae': k,
            'num_time_slices': 50,
            'expansion_mode': 'suzuki',
            'expansion_order': 2,
            'hermitian_matrix': True,
            'negative_evals': nege,
            'backend' : "local_qasm_simulator",
            #'evo_time': 2*np.pi/4,
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

qpe.init_params(params, matrix)

qc = qpe._compute_eigenvalue()
res = qpe._ret

print("Results:", res["measurements"][:10])
print("Evolution time 2Pi/t:", 2*np.pi/res["evo_time"])

def plot_res_and_theory(res):
    def theory(y, w, k, n, t):
        r = np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/
            (1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))
        r[np.isnan(r)] = 2**k
        r = 2**(-2*k-n)*r**2
        r/=sum(r)
        return r

    x = []
    y = []
    for c, _, l in res["measurements"]:
        x += [l]
        y += [c]

    ty = np.arange(0, 2**k, 1)
    data = theory(ty, w.real, k, n, res["evo_time"])

    if nege:
        tx = np.arange(0, 2**k, 1)/2**k
        tx[2**(k-1):] = -(1-tx[2**(k-1):])
        tx *= 2*np.pi/res["evo_time"]
        tx =   np.concatenate((tx[2**(k-1):], tx[:2**(k-1)])) 
        data = np.concatenate((data[2**(k-1):], data[:2**(k-1)])) 
    else:
        tx = np.arange(0, 2**k, 1)/2**k
        tx *= 2*np.pi/res["evo_time"]

    plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
    plt.plot(tx, data, "r")

    plt.show()

plot_res_and_theory(res)

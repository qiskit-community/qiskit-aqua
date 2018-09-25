import scipy
from qiskit import register
from InitialStateTest import IST
import numpy as np
from qiskit import available_backends
import sys
import scipy.sparse as sparse
from scipy.sparse.linalg import inv, eigs, expm, eigsh
from numpy.linalg import eig, eigh, svd
from qiskit_aqua.utils import random_matrix_generator as rmg
from qiskit.tools.qi.qi import state_fidelity
import matplotlib.pyplot as plt

def _gen_sparse_hermitian(n):
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())
    matrix = np.round(matrix + np.identity(n),1)
    matrix[8][n-4] = 0
    matrix[n-4][8] = 0
    matrix[n-7][n-4] = 0
    matrix[n-4][n-7] = 0
    matrix[n-12][n-2] = 0
    matrix[n-2][n-12] = 0
    return(matrix)

z = 5
fidadd = 0
test = 0
neg = 0
for a in range(z):
    state_in = IST()
    matrix = rmg.random_hermitian(4, eigrange = [0, 10], sparsity = 0.5)
    #print(matrix)
    w,v = np.linalg.eig(matrix)
    maxeig = np.argmax(w)
    print(w)
    invec = v[maxeig]
    print(invec)
    n = int(np.log2(len(invec)))
    params = {
    'algorithm': {
            'name': 'State_Test',
            'num_ancillae': 3,
            'backend': 'local_qasm_simulator'
    },
    "initial_state": {
        "name": "CUSTOM",
        "state_vector": invec
    }}

    state_in.init_params(params, n)

    res, res2, result = state_in.run()
    #statevector = result.get_data()['statevector']
    #print(statevector)
    #print(res)
    keys = [k for k in res[0]]
    #print(invec)
    #print(np.absolute(invec))
    #print("----------------------")
    #print(np.absolute(res))
    diff = 0
    for i in range(len(keys)):
        #print(i, res[0][keys[i]])
        diff += np.sqrt(res[0][keys[i]][0]**2 + res[0][keys[i]][1]**2) - np.absolute(invec)[i]        
    print(diff)
    print(keys[0], res[0][keys[0]][0], res[0][keys[0]][1])
    print(invec[0])

    #for i in keys:
    #    print(i, np.sqrt(res[0][i][0]**2 + res[0][i][1]**2))
    #    fid = state_fidelity(invec, statevector)
    #    print(state_fidelity(invec, statevector))
    #    with open("fidelity16.txt", "a+") as f:
    #        f.write("Eigenvector = " + str(invec) + "\n" + "Statevector = " + str(statevector) + "\n" + "Fidelity = " + str(fid) + "\n")
    #    fidadd += fid

#with open("fidelity16.txt", "a+") as f:
#    f.write("Fid = " + str(fidadd) + "Z = " + str(z))
    #print(res)
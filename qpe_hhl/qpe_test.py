from qpe import QPE
from qiskit_aqua import Operator
import scipy
import numpy as np
from qiskit import available_backends

import matplotlib.pyplot as plt

qpe = QPE()
#print(available_backends({'local' : True, 'simulator' : True}))
n = 5
k = 8
w = [-1, 0, 0, 0]
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())

    w, v = np.linalg.eig(matrix)
#matrix = [[1, 2], [0, 3]]
#matrix = np.array(matrix)
#matrix = np.diag([1.5, 2.7, 3.8, 5.1])#10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
w, v = np.linalg.eig(matrix)

#op = Operator(matrix=matrix)
#op._check_representation("paulis")
#op._simplify_paulis()
#paulis = op.paulis
#d = []
#for fac, paul in paulis:
#    d += [[fac, paul.to_label()]]
#print(d)


print(matrix)
#print(np.amax(abs(v))/np.amin(abs(v)))
#print(v.real)
print("eigenvalues ", w)

def fitfun(y, w, k, n, t):
    return 2**(-2*k-n)*np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/(1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))**2

invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]

params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': k,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'hermitian_matrix': True
        #'evo_time': 2*np.pi/4,#
        #'use_basis_gates': False,
},
"iqft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": invec#[1/2**0.5,1/2**0.5]
}}

qpe.init_params(params, matrix)

res = qpe.run()

print(res["measurements"][:10])
print(2*np.pi/res["evo_time"])
print(qpe._use_basis_gates)
x = []
y = []
for c, _, l in res["measurements"]:
    x += [l]
    y += [c]

tx = np.arange(0, 2**k, 0.1)/2**k
tx *= 2*np.pi/res["evo_time"]
ty = np.arange(0, 2**k, 0.1)

plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
plt.plot(tx, 1024*fitfun(ty, w.real, k, n, res["evo_time"]), "r")

plt.show()

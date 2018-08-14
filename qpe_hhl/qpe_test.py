from qpe import QPE
from qiskit_aqua import Operator
import scipy
import numpy as np

import matplotlib.pyplot as plt

qpe = QPE()
n = 4
<<<<<<< HEAD
k = 8
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())

    w, v = np.linalg.eig(matrix)

matrix = np.diag([1.5, 2.7, 3.8, 5.1])#10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
w, v = np.linalg.eig(matrix)

=======
v = np.array([-1, -1, -1, -1])
print(v)

while np.amin(v) <= 0:
    matrix = scipy.sparse.random(n, n, density = 0.4, data_rvs = np.random.randn)
    matrix = np.round(5 *( matrix.A + np.matrix.getH(matrix.A)), 3)
    v = np.linalg.eigvals(matrix)
    op = Operator(matrix=matrix)
    

#matrix = np.array([[3, 0, 0 , 0], [0, 2.7, 0, 0], [0, 0, 11.8, 0], [0, 0, 0, 5.3]])
>>>>>>> Isabel/sparse
op = Operator(matrix=matrix)
op._check_representation("paulis")
op._simplify_paulis()
paulis = op.paulis
d = []
for fac, paul in paulis:
    d += [[fac, paul.to_label()]]
print(d)


print(matrix)
print(np.amax(abs(v))/np.amin(abs(v)))
print(v.real)
print(v)

<<<<<<< HEAD
def fitfun(y, w, k, n, t):
    return 2**(-2*k-n)*np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/(1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))**2

invec = sum([v[:,i] for i in range(n)])
=======
invec = v
>>>>>>> Isabel/sparse
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]

params = {
'algorithm': {
        'name': 'QPE',
<<<<<<< HEAD
        'num_ancillae': k,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        #'evo_time': 2*np.pi/4,
        #'use_basis_gates': False,
=======
        'num_ancillae': 4,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 3,
        #'evo_time': np.pi/2
>>>>>>> Isabel/sparse
},
"iqft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": invec#[1/2**0.5,1/2**0.5]
}}

qpe.init_params(params, op)

res = qpe.run()

print(res["measurements"][:10])
print(2*np.pi/res["evo_time"])
print(qpe._use_basis_gates)
x = []
y = []
for c, _, l in res["measurements"]:
    x += [l]
    y += [c]

plt.bar(x, y, width=0.1)
plt.show()

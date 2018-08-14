from qpe import QPE
from qiskit_aqua import Operator
from qiskit import register
import Qconfig
import numpy as np

import matplotlib.pyplot as plt

register(Qconfig.APItoken, Qconfig.config["url"])

qpe = QPE()

w = [-1, -1]
n = 4
k = 7
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())

    w, v = np.linalg.eig(matrix)

#matrix = np.diag(10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
w, v = np.linalg.eig(matrix)

op = Operator(matrix=matrix)
op._check_representation("paulis")
op._simplify_paulis()
paulis = op.paulis
d = []
for fac, paul in paulis:
    d += [[fac, paul.to_label()]]
print(d)


print(matrix)
print(max(abs(w))/min(abs(w)))
print(w.real)
print(v)

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
        'num_time_slices': 5,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        #'evo_time': 2*np.pi/8,
        #'use_basis_gates': False,
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

tx = np.arange(0, 2**k, 1)/2**k
tx *= 2*np.pi/res["evo_time"]
ty = np.arange(0, 2**k, 1)

plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
plt.plot(tx, 1024*fitfun(ty, w.real, k, n, res["evo_time"]), "r")

plt.show()

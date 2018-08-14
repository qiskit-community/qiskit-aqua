from qpe import QPE
from qiskit_aqua import Operator

import numpy as np

import matplotlib.pyplot as plt

qpe = QPE()

w = [-1, -1]
n = 4
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())

    op = Operator(matrix=matrix)
    w, v = np.linalg.eig(matrix)

matrix = np.array([[3, 0, 0 , 0], [0, 2.7, 0, 0], [0, 0, 11.8, 0], [0, 0, 0, 5.3]])
op = Operator(matrix=matrix)

print(matrix)
print(max(abs(w))/min(abs(w)))
print(w.real)
print(v)

invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]

params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': 4,
        'num_time_slices': 50,
        'expansion_mode': 'suzuki',
        'expansion_order': 4,
        #'evo_time': np.pi/2
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

print(res["measurements"])
x = []
y = []
for c, k, l in res["measurements"]:
    x += [l]
    y += [c]

plt.bar(x, y, width=0.1)
plt.show()

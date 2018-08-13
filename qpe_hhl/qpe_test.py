from qpe import QPE
from qiskit_aqua import Operator

import numpy as np

import matplotlib.pyplot as plt

qpe = QPE()

w = [-1, -1]

while min(w) <= 0:
    matrix = np.random.random([2, 2])+1j*np.random.random([2, 2])
    matrix = 4*(matrix+matrix.T.conj())

    op = Operator(matrix=matrix)
    w, v = np.linalg.eig(matrix)
print(matrix)
print(max(abs(w))/min(abs(w)))
print(w.real)

invec = v[0]+v[1]
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))


params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': 9,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 1,
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

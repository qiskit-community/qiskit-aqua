from qpe import QPE
from qiskit_aqua import Operator
import scipy
import numpy as np

import matplotlib.pyplot as plt

qpe = QPE()
n = 4
v = np.array([-1, -1, -1, -1])
print(v)

while np.amin(v) <= 0:
    matrix = scipy.sparse.random(n, n, density = 0.4, data_rvs = np.random.randn)
    matrix = np.round(5 *( matrix.A + np.matrix.getH(matrix.A)), 3)
    v = np.linalg.eigvals(matrix)
    op = Operator(matrix=matrix)
    

#matrix = np.array([[3, 0, 0 , 0], [0, 2.7, 0, 0], [0, 0, 11.8, 0], [0, 0, 0, 5.3]])
op = Operator(matrix=matrix)

print(matrix)
print(np.amax(abs(v))/np.amin(abs(v)))
print(v.real)
print(v)

invec = v
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]

params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': 4,
        'num_time_slices': 4,
        'expansion_mode': 'suzuki',
        'expansion_order': 3,
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

from qpe import QPE
from qiskit_aqua import Operator

import numpy as np

import matplotlib.pyplot as plt

qpe = QPE()

op = Operator(matrix=[[2, -1], [-1, 2]])

params = {
'algorithm': {
        'name': 'QPE',
        'num_ancillae': 5,
        'num_time_slices': 2,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        #'evo_time': np.pi/2
},
"iqft": {
    "name": "STANDARD"
},
"initial_state": {
    "name": "CUSTOM",
    "state_vector": [1/2**0.5, -1/2**0.5]
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

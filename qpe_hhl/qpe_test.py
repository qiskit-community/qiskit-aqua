from qpe import QPE
from qiskit_aqua import Operator
from qiskit import register
import Qconfig
import numpy as np

import matplotlib.pyplot as plt

register(Qconfig.APItoken, Qconfig.config["url"])

qpe = QPE()

w = [-1, -1]
<<<<<<< HEAD
n = 4           #matrix dimensions

||||||| merged common ancestors
n = 4
k = 8
=======
n = 4
k = 7
>>>>>>> 918414ef15dbf71aec513a4910739308c1484a11
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())
    op = Operator(matrix=matrix)
    w, v = np.linalg.eig(matrix)
<<<<<<< HEAD
    
matrix = np.diag([2,4,16,8])
||||||| merged common ancestors

matrix = np.diag([1.5, 2.7, 3.8, 5.1])#10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
w, v = np.linalg.eig(matrix)

=======

#matrix = np.diag(10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
w, v = np.linalg.eig(matrix)

>>>>>>> 918414ef15dbf71aec513a4910739308c1484a11
op = Operator(matrix=matrix)
w, v = np.linalg.eig(matrix)

print('matrix: \n', matrix)
print('condition number \n', max(abs(w))/min(abs(w)))
print('eigenvalues (Re):', w.real)
print('eigenvectors: \n', v)
#%%
invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]
#%%
params = {
'algorithm': {
        'name': 'QPE',
<<<<<<< HEAD
        'num_ancillae':4 ,
        'num_time_slices': 10,
||||||| merged common ancestors
        'num_ancillae': k,
        'num_time_slices': 10,
=======
        'num_ancillae': k,
        'num_time_slices': 5,
>>>>>>> 918414ef15dbf71aec513a4910739308c1484a11
        'expansion_mode': 'suzuki',
<<<<<<< HEAD
        'expansion_order': 3,
        #'evo_time': np.pi/2
||||||| merged common ancestors
        'expansion_order': 2,
        #'evo_time': 2*np.pi/4,
        #'use_basis_gates': False,
=======
        'expansion_order': 2,
        #'evo_time': 2*np.pi/8,
        #'use_basis_gates': False,
>>>>>>> 918414ef15dbf71aec513a4910739308c1484a11
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

<<<<<<< HEAD
plt.bar(x, y, width=0.1)
||||||| merged common ancestors
tx = np.arange(0, 2**k, 0.1)/2**k
tx *= 2*np.pi/res["evo_time"]
ty = np.arange(0, 2**k, 0.1)

plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
plt.plot(tx, 1024*fitfun(ty, w.real, k, n, res["evo_time"]), "r")

=======
tx = np.arange(0, 2**k, 1)/2**k
tx *= 2*np.pi/res["evo_time"]
ty = np.arange(0, 2**k, 1)

plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
plt.plot(tx, 1024*fitfun(ty, w.real, k, n, res["evo_time"]), "r")

>>>>>>> 918414ef15dbf71aec513a4910739308c1484a11
plt.show()

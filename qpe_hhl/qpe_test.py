from qpe import QPE
from qiskit_aqua import Operator
import scipy
from qiskit import register
<<<<<<< HEAD
import Qconfig
||||||| merged common ancestors
#import Qconfig
=======
>>>>>>> David/sparse
import numpy as np
from qiskit import available_backends
import sys
sys.path.append("..")
from matrix_gen import gen_matrix

import matplotlib.pyplot as plt
<<<<<<< HEAD
from qiskit.tools.qi.qi import state_fidelity

register(Qconfig.APItoken, Qconfig.config["url"])

try:
    import sys
    sys.path.append("~/workspace/") # go to parent dir
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except Exception as e:
    print(e)
    qx_config = {
        "APItoken":"bad8fd2aba4b1154108dec4b307471b8c20f32afe6b98e59b723f29c0bfc455d4b19e7783ce8d60cd52369909a15349d0d571d1246dedc43ffc21e03ca13a07a",
        "url":"https://quantumexperience.ng.bluemix.net/api"}
register(qx_config['APItoken'], qx_config['url'])

backend = 'ibmqx5'
||||||| merged common ancestors
try:
    import sys
    sys.path.append("~/workspace/") # go to parent dir
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
except Exception as e:
    print(e)
    qx_config = {
        "APItoken":"bad8fd2aba4b1154108dec4b307471b8c20f32afe6b98e59b723f29c0bfc455d4b19e7783ce8d60cd52369909a15349d0d571d1246dedc43ffc21e03ca13a07a",
        "url":"https://quantumexperience.ng.bluemix.net/api"}
register(qx_config['APItoken'], qx_config['url'])

backend = 'ibmqx5'
=======

>>>>>>> David/sparse
qpe = QPE()
<<<<<<< HEAD

#print(available_backends({'local' : True, 'simulator' : True}))
hermitian_matrix = True
||||||| merged common ancestors
#print(available_backends({'local' : True, 'simulator' : True}))
hermitian_matrix = True
=======
>>>>>>> David/sparse
n = 2
<<<<<<< HEAD
k = 3
w = [-1, 0, 0, 0]

while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())
    matrix = np.round(matrix + np.identity(n),1)
    w, v = np.linalg.eig(matrix)   

#matrix = [[1, 2], [0, 3]]
#matrix = np.array(matrix)
#matrix = np.diag([1.5, 2.7, 3.8, 5.1])#10*np.random.random(4))
#matrix = np.diag(10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
#w, v = np.linalg.eig(matrix)
if not hermitian_matrix:
    singval = scipy.linalg.svd(matrix, compute_uv = False)
#op = Operator(matrix=matrix)
#op._check_representation("paulis")
#op._simplify_paulis()
#paulis = op.paulis
#d = []
#for fac, paul in paulis:
#    d += [[fac, paul.to_label()]]
#print(d)
||||||| merged common ancestors
k = 3
w = [-1, 0, 0, 0]
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())
    matrix = np.round(matrix + np.identity(n),1)
    w, v = np.linalg.eig(matrix)   


#matrix = [[1, 2], [0, 3]]
#matrix = np.array(matrix)
#matrix = np.diag([1.5, 2.7, 3.8, 5.1])#10*np.random.random(4))
#matrix = np.diag(10*np.random.random(4))
#matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]])
#w, v = np.linalg.eig(matrix)
if not hermitian_matrix:
    singval = scipy.linalg.svd(matrix, compute_uv = False)
#op = Operator(matrix=matrix)
#op._check_representation("paulis")
#op._simplify_paulis()
#paulis = op.paulis
#d = []
#for fac, paul in paulis:
#    d += [[fac, paul.to_label()]]
#print(d)
=======
k = 9
nege = True
>>>>>>> David/sparse

<<<<<<< HEAD
||||||| merged common ancestors

=======
matrix = gen_matrix(n, eigrange=[-5, 5], sparsity=0.6)
#matrix = np.diag([-1.5, 1])
#np.save("mat.npy", matrix)
#matrix = np.load("mat.npy")
w, v = np.linalg.eigh(matrix) 

>>>>>>> David/sparse
print(matrix)
<<<<<<< HEAD
#print(np.amax(abs(v))/np.amin(abs(v)))
#print(v.real)
print("eigenvalues ", w)
#print("singular values", singval)


def fitfun(y, w, k, n, t):
    return 2**(-2*k-n)*np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/(1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))**2
||||||| merged common ancestors
#print(np.amax(abs(v))/np.amin(abs(v)))
#print(v.real)
print("eigenvalues ", w)
#print("singular values", singval)

def fitfun(y, w, k, n, t):
    return 2**(-2*k-n)*np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/(1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))**2
=======
print("Eigenvalues:", w)
>>>>>>> David/sparse

invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))

#invec =v[:,0]
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

<<<<<<< HEAD


res = qpe.run()
||||||| merged common ancestors
res = qpe.run()
=======
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
>>>>>>> David/sparse

<<<<<<< HEAD
print(res["measurements"][:10])
print(2*np.pi/res["evo_time"])
print(qpe._use_basis_gates)
x = []
y = []

for c, _, l in res["measurements"]:
    x += [l]
    y += [c]
||||||| merged common ancestors
print(res["measurements"][:10])
print(2*np.pi/res["evo_time"])
print(qpe._use_basis_gates)
x = []
y = []
for c, _, l in res["measurements"]:
    x += [l]
    y += [c]
=======
    if nege:
        tx = np.arange(0, 2**k, 1)/2**k
        tx[2**(k-1):] = -(1-tx[2**(k-1):])
        tx *= 2*np.pi/res["evo_time"]
        tx =   np.concatenate((tx[2**(k-1):], tx[:2**(k-1)])) 
        data = np.concatenate((data[2**(k-1):], data[:2**(k-1)])) 
    else:
        tx = np.arange(0, 2**k, 1)/2**k
        tx *= 2*np.pi/res["evo_time"]
>>>>>>> David/sparse

    plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
    plt.plot(tx, data, "r")

    plt.show()

<<<<<<< HEAD
plt.show()

||||||| merged common ancestors
plt.show()
=======
plot_res_and_theory(res)
>>>>>>> David/sparse

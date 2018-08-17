from qpe import QPE
from qiskit_aqua import Operator
from qiskit import register
#import Qconfig
import numpy as np

import matplotlib.pyplot as plt
from qiskit.tools.qi.qi import state_fidelity
#register(Qconfig.APItoken, Qconfig.config["url"])

qpe = QPE()

w = [-1, -1]
n = 2
k = 4
while min(w) <= 0:
    matrix = np.random.random([n, n])+1j*np.random.random([n, n])
    matrix = 4*(matrix+matrix.T.conj())

    w, v = np.linalg.eig(matrix)

#matrix = np.diag([3,1])

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


print('Operator matrix:', matrix)
print(max(abs(w))/min(abs(w)))
print('Eigenvalues:',w.real)
print(v)

def fitfun(y, w, k, n, t):
    return 2**(-2*k-n)*np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/(1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))**2

invec = sum([v[:,i] for i in range(n)])
invec /= np.sqrt(invec.dot(invec.conj()))
#op = Operator(matrix=1/4*np.array([[15, 9, 5, -3], [9, 15, 3, -5], [5, 3, 15, -9], [-3, -5, -9, 15]]))
#invec = [0,0,0,1]

#invec =v[:,0]
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


test_flag = False
qpe.init_params(params, op,test_flag )
res, test_results = qpe.run()

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

#%%
#input vector:


def test_fidelity(k, invec)   : 
    #generating quantum state with qpe:
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
    test_flag = True
    qpe_t = QPE()
    qpe_t.init_params(params, op, test_flag)
    res2, test_results2 = qpe_t.run()
    
    print('State vector QPE algorithm:\n', np.round(test_results2.get_statevector(),3))
    qvec = test_results2.get_statevector()
    t = res2["evo_time"]
    
    
    # generating theoretical state vector
    coeff_l = np.zeros((2**k,n), dtype = complex)
    
    for l in np.arange(2**k):
        for m in np.arange(2**k):       
            coeff_l[l] = coeff_l[l] + (1/(2**k))*np.exp((-2.j*np.pi*m*l/(2**k)))* np.exp((1.j*w*t*m))
            #print(l, m, coeff_l[l])
            
    #alternative method for verification if coeff_l is right
    # =============================================================================
    # q = np.zeros(2**k, dtype = complex)
    # wi = w[1]
    # for l in np.arange(2**k):
    #     q[l] = 2**(-k)*sum(np.exp(-2j*np.pi*x*(2**(-k))*l)*np.exp(1j*x*wi*t) for x in np.arange(2**k))
    #     
    # =============================================================================
    # it's the same
              
    #cvec = np.array(coeff_l)
    #c = np.outer(invec, cvec).reshape((np.shape(np.outer(invec, cvec))[1],np.shape(np.outer(invec, cvec))[0]))    
    #C = np.sum(c,axis = 1)#/np.sqrt(n)#/np.linalg.norm(np.sum(c,axis = 1))
    #C = (np.matrix(c)*np.matrix(invec).T).round(3)
    
    C1 = (coeff_l.round(3)*invec).reshape((n*2**k, 1), order = 'F')
    
    #generate density matrices:
    rho_q = np.outer(qvec.round(3), qvec.round(3))
    #rho_t = 1/np.sqrt(2)* (np.outer(c[:, 0], c[:, 0]) + np.outer(c[:, 1], c[:, 1]))
    rho_t = np.outer(C1,C1)
    
    print('Input state', invec.round(3) )
    print('Theoretical state vector: \n', np.round(C1,3))
    #print('Fidelity:\n', state_fidelity(qvec, np.array(C).reshape(len(C))))
    print('Fidelity (state vectors):\n', state_fidelity(qvec, C1.reshape(len(C1))).round(3))
    print('Fidelity (density matrices):\n', state_fidelity(rho_q, rho_t).round(3))
    
    #np.tensordot(np.round(coeff_l, 3), invec, axes = 0).reshape((8, 2))
    # =============================================================================
    # state_fidelity(qvec,c[:,0])
    # state_fidelity(qvec,c[:,0]*np.sqrt(2)) # this gives 1 for invec = v[0]
    # =============================================================================
    return

for i in range(n):
    test_fidelity(k, v[i])
    
test_fidelity(k, invec)
from qpe import QPE
from qiskit_aqua import Operator
import numpy as np
from qiskit.tools.qi.qi import state_fidelity

def qpe_test_fidelity(n, k, matrix):
    """
    n - siye of the system
    k - number of ancilla qubits
    matrix - operator matrix, size nxn
    """
    
    w, v = np.linalg.eig(matrix)

    op = Operator(matrix=matrix)
    op._check_representation("paulis")
    op._simplify_paulis()
    paulis = op.paulis
    d = []
    
    for fac, paul in paulis:
        d += [[fac, paul.to_label()]]
    
    if False:    
        print(d)
        print('Operator matrix:', matrix)
        print(max(abs(w))/min(abs(w)))
        print('Eigenvalues:',w.real)
        print(v)
    
    test_results = ['Eigevector', 'Eigenvalue', 'Fidelity']
    
    for i2 in range(n):   
        invec = v[:,i2]    
        #generating quantum state with qpe:
        params = {
        'algorithm': {
                'name': 'QPE',
                'num_ancillae': k,
                'num_time_slices': 5,
                'expansion_mode': 'suzuki',
                'expansion_order': 4,
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
        qvec = test_results2.get_statevector()    
        t = res2["evo_time"]
            
        # generating theoretical state vector - calculating the coefficients for the basis vector:
        coeff_l = np.zeros((2**k,1), dtype = complex)
        
        for l in np.arange(2**k):
            for m in np.arange(2**k):       
                coeff_l[l] = coeff_l[l] + np.exp((-2.j*np.pi*m*l/(2**k)))* np.exp((1.j*w[i2]*t*m))/(2**k)
                #print(l, m, (np.exp((-2.j*np.pi*m*l/(2.**k)))* np.exp((1.j*w[i]*t*m))).round(3))
                
        # Mapping that accounts for tensor product in reversed order:        
        tmap = np.zeros((2**k, 2**k))
        np.tensordot([1, 0],[0,1], axes = 0).reshape((1, 4))
        
        for i in range(2**k):
            bv  = format(i, 'b').zfill(8)
            bv = bv[-k:]            
            if int(bv[-1]) == 1:
                s = [0,1]
            else: 
                s = [1, 0]           
            for j in range(2, k+1):
                    if int(bv[-j]) == 1:
                        s = np.tensordot(s,[0,1], axes = 0).reshape((2**j))
                    else: 
                        s = np.tensordot(s, [1,0], axes = 0).reshape((2**j)) 
            tmap[:,i] = s
        
        #Generating theoretical vector mapped onto QPE state vector basis (with reversed tensor product)
        coeff_l_swap = (np.matrix(tmap)*coeff_l)   
        C1 = np.tensordot(invec, coeff_l_swap.round(3), axes = 0).reshape((n*2**k,1))
        
        if False:
            print('Eigenstate', i2, '\t', invec)
            print('Input state', invec.round(3) )
            print('Theoretical state vector: \n', np.round(C1,3))
            print('QPE  state vector: \n', np.round(C1,3))
            print('Fidelity (state vectors): \t', state_fidelity(qvec, C1.reshape(len(C1))))
    
        test_results = np.vstack((test_results, [invec, w[i2], state_fidelity(qvec, C1.reshape(len(C1)))]))
    
    return test_results


#sample run:
if True:    
    k = 2
    matrix = np.diag([3.5,1, 2, 6])
    n = len(matrix)
    print(qpe_test_fidelity(n, k, matrix))


#Run tests over parameter grid (k, n):
if False:
    T = ['Test results', k, n]
    for k in np.arange(3, 7):
        for n in [2, 4]: 
            w = [-1, -1]
            while min(w) <= 0:
                matrix = np.random.random([n, n])+1j*np.random.random([n, n])
                matrix = 4*(matrix+matrix.T.conj())
                w, v = np.linalg.eig(matrix)
            print('Matrix:', matrix)
            #matrix = np.random.rand(n, n)
            #matrix = matrix + matrix.T
            Ttemp = qpe_test_fidelity(n, k, matrix)
            T = np.vstack((T, Ttemp))
            print(Ttemp)

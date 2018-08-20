from qpe import QPE
#from qiskit_aqua import Operator
import numpy as np
from qiskit.tools.qi.qi import state_fidelity
from qiskit import execute

def qpe_test_fidelity(n, k, matrix, num_time_slices, expansion_mode, expansion_order):
    """
    n - size of the system
    k - number of ancilla qubits
    matrix - operator matrix, size nxn
    """       
    hermitian_matrix = True    
    #k = 2
    #matrix = np.diag([3.5,1, 2, 6])
    #n = len(matrix)
    w, v = np.linalg.eig(matrix)
    test_results = np.empty((n, 5)) #['Eigevector', 'Eigenvalue', 'Fidelity']
    
    for i2 in range(0,n):   
        invec = v[:,i2]   
        backend = 'local_statevector_simulator'
        #invec = v[0]
        #num_time_slices = 5
        #expansion_mode = 'suzuki'
        #expansion_order = 2
        #k = 3
        #hermitian_matrix = True
        backend = 'local_statevector_simulator'
        
        params = {
                'algorithm': {
                        'name': 'QPE',
                        'num_ancillae': k,
                        'num_time_slices': num_time_slices,
                        'expansion_mode': expansion_mode,
                        'expansion_order': expansion_order,
                        'hermitian_matrix': hermitian_matrix,
                        'backend' : backend
     
                },
                "iqft": {
                    "name": "STANDARD"
                },
                "initial_state": {
                    "name": "CUSTOM",
                    "state_vector": invec#[1/2**0.5,1/2**0.5]
                }}
        #generating quantum state with qpe:        
        qpe = QPE()
        qpe.init_params(params, matrix)    
        circuit, t = qpe._setup_qpe(measure = False)
        res =  execute(circuit, backend = backend)
        qvec = res.result().get_statevector() 
            
        # generating theoretical state vector - 
        #calculating the coefficients for the basis vector:
        coeff_l = np.zeros((2**k,1), dtype = complex)
        
        for l in np.arange(2**k):
            for m in np.arange(2**k):       
                coeff_l[l] = coeff_l[l] + np.exp((-2.j*np.pi*m*l/(2**k)))*np.exp((1.j*w[i2]*t*m))/(2**k)
                
                
        # Mapping that accounts for tensor product in reversed order:        
        tmap = np.zeros((2**k, 2**k))
        #np.tensordot([1, 0],[0,1], axes = 0).reshape((1, 4))
        
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
        
        #Generating theoretical vector mapped onto QPE state vector basis 
        #with reversed tensor product)
        coeff_l_swap = (np.matrix(tmap)*coeff_l)   
        C1 = np.tensordot(invec, coeff_l_swap.round(3), axes = 0).reshape((n*2**k,1))
        
        if False:
            print('Eigenstate', i2, '\t', invec)
            print('Input state', invec.round(3) )
            print('Theoretical state vector: \n', np.round(C1,3))
            print('QPE  state vector: \n', np.round(C1,3))
            print('Fidelity (state vectors): \t', state_fidelity(qvec, C1.reshape(len(C1))))
    
        
        test_results[i2, :] = np.array([n, k, num_time_slices, expansion_order,
                                state_fidelity(qvec, C1.reshape(len(C1)))])
    
    return test_results

#%%
#dry run:
if False:    
    k = 2
    matrix = np.diag([3.5,1, 2, 6])
    n = len(matrix)
    w, v = np.linalg.eig(matrix)
    
    invec = v[0]
    num_time_slices = 5
    expansion_mode = 'suzuki'
    expansion_order = 2
    k = 3
    hermitian_matrix = True
    backend = 'local_statevector_simulator'
    
    params = {
            'algorithm': {
                    'name': 'QPE',
                    'num_ancillae': k,
                    'num_time_slices': num_time_slices,
                    'expansion_mode': expansion_mode,
                    'expansion_order': expansion_order,
                    'hermitian_matrix': hermitian_matrix,
                    'backend' : backend
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
    
    
    #print(qpe_test_fidelity(n, k, matrix))
    qpe = QPE()
    qpe.init_params(params, matrix)    
    circuit, t = qpe._setup_qpe(measure = False)
    print('circuit is on !!')
    res =  execute(circuit, backend = backend)
    print('results ready')
    R = res.result().get_data()
    print('data acquired')
    qvec = res.result().get_statevector()
    print('Done!!')    
#%%
test1 = ['n', 'k', 'time slices', 'expansion order', 'Fidelity']

#%%
#Run tests over parameter grid (k, n):
if True:
    for expansion_order in [2, 3, 4]:
        for expansion_mode in ['suzuki']:
            for num_time_slices in [5]:    
                for k in np.arange(3,4):
                    for n in [2]: 
                        w = [-1, -1]
                        while min(w) <= 0:
                            matrix = np.random.random([n, n])+1j*np.random.random([n, n])
                            matrix = 4*(matrix+matrix.T.conj())
                            w, v = np.linalg.eig(matrix)
                            
                        test1 = np.vstack((test1, 
                               qpe_test_fidelity(n, k, matrix, num_time_slices, 
                                                 expansion_mode, expansion_order)))    
                
            
            #print('Matrix:', matrix)
            #matrix = np.random.rand(n, n)
            #matrix = matrix + matrix.T
            #Ttemp = qpe_test_fidelity(n, k, matrix)
            #T = np.vstack((T, Ttemp))
            #print(Ttemp)
            
            

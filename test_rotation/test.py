from qiskit_aqua.algorithms.single_sample.hhl.lookup_rotation import LUP_ROTATION
from qiskit_aqua.algorithms.single_sample.hhl.qpe import QPE
from qiskit.extensions.simulator import snapshot
from qiskit import execute
import numpy as np
#####################
##### standalone test
#####################
#LUP_ROTATION.test_value_range(6,5)


#####################
##### test with QPE
#####################

def test_with_QPE():
    qpe = QPE()
    

    matrix = np.array([[1,0,0],[0,2,0],[0,0,3]])#,
    matrix =np.array( [[1,0,0,0],[0,2,0,0],  [0,0,3,0],[0,0,0,4]])
    matrix = np.array([[2,-1],[-1,2]])
    #gen_matrix(n, eigrange=[-5, 5], sparsity=0.6)

    n = int(matrix.shape[0])
    print(type(n))
    k = 4
    nege = False#True
    
    w, v = np.linalg.eigh(matrix) 

    print([(w[i],v[:,i])for i in range(n)])

    invec = sum([v[:,i] for i in range(n)])
    invec /= np.sqrt(invec.dot(invec.conj()))
    invec =1/np.sqrt(2)*np.array([1,1])
    params = {
    'algorithm': {
        'name': 'QPE',
        'num_ancillae': k,
        'num_time_slices': 1,
        'expansion_mode': 'trotter',#'suzuki',
        'expansion_order': 5,
        'hermitian_matrix': True,
        'negative_evals': nege,
        'backend' : "local_qasm_simulator",
        #'evo_time': 2*np.pi/np.max(matrix),
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
    print(invec)
    qpe.init_params(params, matrix)


    qc = qpe._setup_qpe()
    
    #####################
    ## global phase test
    #####################
    
    #qc.u1(np.pi,qc.regs['comp'])
    #qc.x(qc.regs['comp'])
    #qc.u1(np.pi,qc.regs['comp'])
    #qc.x(qc.regs['comp'])
    #qc.snapshot("1")
    #res = execute(qc,backend="local_qasm_simulator",config={"data":["quantum_state_ket"]},shots=1)
    #res_ = res.result().get_data()["snapshots"]["1"]["quantum_state_ket"]
    #print(res_)
    #print(res.result().get_data()["snapshots"]["1"]["statevector"])
    #return

    
    ev_register = qc.regs['eigs']
    n_ = 3
    params = {
            'algorithm': {
                'reg_size': k,
                'pat_length': n_,
                'subpat_length': int(np.ceil(n_ / 2)),
                'negative_evals': nege,
                'backend': 'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "ZERO",
                "state_vector": []
            },
            "qpe_hhl": {
                "name": "STANDARD",
                "circuit": qc,
                "ev_register":ev_register
            }
        }

    obj = LUP_ROTATION()
    print(obj)
    obj.init_params(params)
    res = obj.run(1)
    print(res)
test_with_QPE()


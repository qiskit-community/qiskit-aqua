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
    k = 4
    nege = False#True
    
    w, v = np.linalg.eigh(matrix) 

    print([(w[i],v[:,i])for i in range(n)])

    invec = sum([v[:,i] for i in range(n)])
    invec /= np.sqrt(invec.dot(invec.conj()))
    invec =-1/np.sqrt(2)*np.array([1,-1])
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
    evo_time = qpe._evo_time
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
                "ev_register":ev_register,
                "evo_time": evo_time
            }
        }

    obj = LUP_ROTATION()
    obj.init_params(params)
    res = obj.run(1)
    print(res)
test_with_QPE()







def test_value_range(k,n):
        backend = 'local_qasm_simulator'#'local_statevector_simulator'
        negative_evals = True
        params = {
            'algorithm': {
                'reg_size':k,
                'pat_length':n,
                'subpat_length':int(np.ceil(n/2)),
                'negative_evals':negative_evals,
                'backend':backend#'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "CUSTOM",
                "state_vector": []
            },
            "qpe_hhl":{
                "name": "ZERO"
            }
        }
        res_dict = {}
        for pattern in itertools.product('01', repeat=k):
            if not '1' in pattern:
                continue
            ####
            t = ['0']*k
            t[-2] = '1'
            t[-1] = '1'
            #if pattern !=tuple(t):
            #    continue
            print(pattern)
            ####
            state_vector = LUP_ROTATION.get_initial_statevector_representation(list(pattern))
            #state_vector = np.zeros(2**k)
            #state_vector[-1] = 1
            if negative_evals:

                num = np.sum([2 ** -(n + 2) for n, i in enumerate(reversed(pattern[:-1])) if i == '1'])
                if pattern[-1] == '1':
                    num *= -1
                if pattern[-1] and not '1' in pattern[:-1]:
                    num = -0.5
            else:
                num = np.sum([2 ** -(n + 1) for n, i in enumerate(reversed(pattern)) if i == '1'])

            params["initial_state"]["state_vector"] = state_vector
            # state_vector = np.zeros(2**k)
            # state_vector[4] = 1

            print(state_vector)
            print("Next pattern:", pattern, "Numeric:",num)
            obj = LUP_ROTATION()
            obj.init_params(params)
            res = obj.run()
            #break
            
            if backend == 'local_statevector_simulator' or 1:
                res_dict.update(res)
                continue
            for d in res:
                if d[1][0] == '1':
                    res_dict.update({float(d[2].split()[-1]): d[0]})
        if backend == 'local_qasm_simulator':
            vals = list(res_dict.keys())
            inverse = [res_dict[_] for _ in vals]
            vals = np.array(vals)
            inverse = np.array(inverse)
            plt.scatter(vals, inverse / 2 ** -k)
            if negative_evals:
                x = np.linspace(-0.5, -2**-k, 1000)
                plt.plot(x, 1 / x)
                x = np.linspace(2**-k, 0.5, 1000)
                plt.plot(x,1/x)
            else:
                plt.plot(np.linspace(2 ** -k, 1, 1000), 1 / np.linspace(2 ** -k, 1, 1000))
            plt.show()
            
def test_stat_error(k, n):
        params = {
            'algorithm': {
                'reg_size': k,
                'pat_length': n,
                'subpat_length': int(np.ceil(n / 2)),
                'negative_evals': False,
                'backend': 'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "CUSTOM",
                "state_vector": []
            },
            "qpe_hhl": {
                "name": "ZERO"
            }
        }
        counts = []
        pattern = ['1'] * k
        state_vector = LUP_ROTATION.get_initial_statevector_representation(list(pattern))
        params["initial_state"]["state_vector"] = state_vector

        for i in range(100):
            obj = LUP_ROTATION()
            obj.init_params(params)
            res = obj.run()
            # break
            for d in res:
                if d[1][0] == '1':
                    counts.append(d[0])
        plt.hist(counts)
        plt.show()

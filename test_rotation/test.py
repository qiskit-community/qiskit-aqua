from qiskit_aqua.algorithms.single_sample.hhl.lookup_rotation import LUP_ROTATION
from qiskit_aqua.algorithms.single_sample.hhl.qpe import QPE
from qiskit.extensions.simulator import snapshot
from qiskit_aqua.utils import random_hermitian
from qiskit import execute
import matplotlib.pyplot as plt
import numpy as np
import itertools

def get_statevector_representation(bitpattern):
    state_vector = None
    for bit in bitpattern:
        vec = np.array([0,1]) if bit=='1' else np.array([1,0])
        if state_vector is None:
            state_vector = vec
            continue
        state_vector = np.tensordot(state_vector,vec,axes=0)
    return state_vector.flatten()

def test_value_range(k,n):
        backend = 'local_qasm_simulator'#'local_statevector_simulator'
        negative_evals = False#True
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
        
        x = []
        y = []
            
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
            state_vector = get_statevector_representation(list(pattern))
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
            res = obj._execute_rotation(1)
            anc_1 = [(d[1],d[2]) for d in res if d[0]=='Anc 1']
            for i in anc_1:
                x.append(i[0])
                if obj._evo_time is not None:
                    y.append(i[1]*2**k)
                else:
                    y.append(i[1]*2**k)
        
        
        if negative_evals:
            x_ = np.append(np.linspace(-0.5,-2**-k,500),np.linspace(2**-k,0.5,500))
        else:
            x_ = np.linspace(0,1,1000)
        plt.scatter(x,y)
        plt.plot(x_,1/x_)
        plt.show()
        return
  

#####################
##### test with QPE
#####################

def test_with_QPE(config):
    config_implemented = [
         'QPE_standalone',
         'QPE_globalphase_test',
         'QPE_ROT',
        ]
    test_config = config['test']
    for setting in test_config:
        if setting not in config_implemented:
            raise ValueError("Invalid configuration for test function: {}".format(setting))
    #################
    ## initialize QPE
    #################
    qpe = QPE()

    qpe_config = config['QPE']
    qpe_param =qpe_config['param']
    if 'matrix' not in list(qpe_config.keys()):
        if any([val not in list(qpe_config.keys())
                for val in ['EVmin','EVmax','N','sparsity']]):
               raise ValueError("Missing parameter for matrix generation")
               
        print("Generating random matrix")
        N = qpe_config['N']
        EVmin = qpe_config['EVmin']
        EVmax = qpe_config['EVmax']
        sparsity = qpe_config['sparsity']
        #generate matrix
        #matrix = np.array([[1,0,2],[0,2,0],[2,0,3]])#,
        #matrix =np.array( [[1,0,0.2,0],[0,2,0,0],  [0,0,3,0],[0,0.2,0,4]])
        #matrix = np.array([[2,-1],[-1,2]])
        matrix = random_hermitian(N, eigrange=[EVmin,EVmax], sparsity=sparsity)
        
    n = int(matrix.shape[0])
    # size of EV register
    k = 6
    w, v = np.linalg.eigh(matrix) 
    if any([w[i]<0 for i in range(n)]):
        nege = True
        print("Negative EV present")
    else:
        nege = False
        print("Only positive EV present")
        
    #set explicit pos / neg EV
    #nege = True
    qpe_param['algorithm']['negative_evals'] = nege
    print("Matrix has Eigenvalues/vector:")
    print('\n\n'.join([str(w[i])+':'+'('+' '.join(map(str,np.round(v[:,i],4).tolist()))+' )'for i in range(n)]))
    print("###################################")
    if 'state_vector' not in list(qpe_param.keys()):
        invec_  = sum([v[:,i] for i in range(n)])
        invec = invec_ / np.sqrt(invec_.dot(invec_.conj()))
        #invec = v[:,1]
        qpe_param['initial_state'].update({'state_vector': invec})
    print("Running test with input vector: {}".format(qpe_param['initial_state']['state_vector']))
    print("###################################")
    print(qpe_param)           
    qpe.init_params(qpe_param, matrix)
    
    qc = qpe._setup_qpe()
    print("QPE set up! Number of atomic gates: {}".format(qc.number_atomic_gates()))
    print("###################################")

    ##################
    ## standalone QPE
    ##################
    if 'QPE_standalone' in test_config:
    
        qc.snapshot("0")
        result = execute(qc, backend="local_qasm_simulator",
                     shots=1,config={
                         "data":["hide_statevector","quantum_state_ket"]}).result()

        qpe_res = result.get_data()["snapshots"]["0"]["quantum_state_ket"][0]
        x = []
        y = []
        for d in qpe_res.keys():
                    if nege:
                        num = sum([2 ** -(i + 2)
                            for i, e in enumerate(reversed(d.split()[-1][:-1])) if e == "1"])
                        if d.split()[-1][-1] == '1':
                            num *= -1
                            if '1' not in d.split()[-1][:-1]:
                                num = -0.5
                    else:
                        num = sum([2 ** -(i + 1)
                            for i, e in enumerate(reversed(d.split()[-1])) if e == "1"])
                    x.append(num*(2*np.pi)/qpe._evo_time)
                    y.append(np.sqrt(qpe_res[d][0]**2+qpe_res[d][1]**2))
        plt.scatter(x,y,label='after qpe')
        plt.scatter(x,np.array(y)/np.array(x)*(2*np.pi)/qpe._evo_time*2**-k,label='after rot')
        plt.legend(loc='best')
        plt.show()
        print(qpe_res)
        return

    #####################
    ## global phase test
    #####################
    if 'QPE_globalphase_test' in test_config:
        qc.u1(np.pi,qc.regs['comp'])
        qc.x(qc.regs['comp'])
        qc.u1(np.pi,qc.regs['comp'])
        qc.x(qc.regs['comp'])
        qc.snapshot("1")
        res = execute(qc,backend="local_qasm_simulator",config={"data":["quantum_state_ket"]},shots=1)
        res_ = res.result().get_data()["snapshots"]["1"]["quantum_state_ket"]
        print(res_)
        print(res.result().get_data()["snapshots"]["1"]["statevector"])
        return
    
    
    ###################
    ## with rotation
    ###################
    if 'QPE_ROT' in test_config:
        ev_register = qc.regs['eigs']
        evo_time = qpe._evo_time
        rot_param = config["ROT"]
        rot_class = config["ROT_CLASS"]
        k = qpe_param['algorithm']['num_ancillae']
        n_ = min(k-1,5)
        
        norm_factor = 1
        obj = rot_class()#LUP_ROTATION()

        #add for other rots
        if isinstance(obj,LUP_ROTATION):
               rot_param['qpe_hhl']["ev_register"] = ev_register
               rot_param['qpe_hhl']["evo_time"] = evo_time
               rot_param['qpe_hhl']["circuit"] = qc
               rot_param['algorithm']['reg_size']= k
               rot_param['algorithm']['pat_length']= int(np.ceil(n_/2))
               rot_param['algorithm']['subpat_length']=n_- int(np.ceil(n_ / 2))
               rot_param['algorithm']['negative_evals']= nege
        print("initializing with parameter:",rot_param)
        obj.init_params(rot_param)
        res = obj._execute_rotation(1)
        print("Total circuit length was: {}".format(qc.number_atomic_gates()))
        print("Of this {} gates for the rotation".format(obj._number_basic_gates))
        print("###################################")

        anc_1 = [(d[1],d[2]) for d in res if d[0]=='Anc 1']
        x = []
        y = []
        for i in anc_1:
            x.append(i[0])
            if obj._evo_time is not None:
                y.append(i[1]/2/np.pi*obj._evo_time*2**k*norm_factor)
            else:
                y.append(i[1]*2**k*norm_factor)
        plt.hist(np.abs(y))
        plt.show()
        plt.hist(np.abs(x))
        plt.show()
        plt.scatter(x,y)
        plt.show()
    
        print(res)
        return

def run_rotation_test():
    qpe_param = {
    'algorithm': {
        'name': 'QPE',
        'num_ancillae': 6,
        'num_time_slices': 10,
        'expansion_mode': 'suzuki',
        'expansion_order': 3,
        'hermitian_matrix': True,
        'backend' : "local_qasm_simulator",
        #'evo_time': 2*np.pi/np.max(matrix),
        #'use_basis_gates': False,
    },
    "iqft": {
        "name": "STANDARD"
    },
    "initial_state": {
        "name": "CUSTOM",
    }
    }


    rot_param = {
            'algorithm': {
            'backend': 'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "ZERO",
                "state_vector": []
            },
            "qpe_hhl": {
                "name": "STANDARD",
            }
        }
    config = {'test':['QPE_ROT'],
            'QPE':{
                'EVmin':-10,
                'EVmax':10,
                'sparsity':0.5,
                'N':6,
                'param':qpe_param
                },
            'ROT':rot_param,
              'ROT_CLASS':LUP_ROTATION,
            }
    test_with_QPE(config)
if __name__=='__main__':
    run_rotation_test()
    #test_value_range(7,3)







"""            
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
"""

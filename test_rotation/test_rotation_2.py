from qiskit_aqua import get_reciprocal_instance, get_initial_state_instance, get_eigs_instance
from qiskit.extensions.simulator import snapshot
from qiskit import QuantumCircuit,QuantumRegister
from qiskit import execute
from qiskit.tools.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np
import itertools
from qiskit_aqua.utils import random_hermitian


def matrix_type():
    t = ['diagonal','hermitian']
    for el in itertools.cycle(t):
        yield el
        
def neg_pos():
    t = ['pos']#,'neg']
    for el in itertools.cycle(t):
        yield el

def sparsity_():
    t = np.arange(10**-1,1,0.1).tolist()
    for el in itertools.cycle(t):
        yield el
        
sparse_gen = sparsity_()
matrix_t_gen = matrix_type()
neg_pos_gen = neg_pos()

def matrix_generator(N):
    mat_t = next(matrix_t_gen)
    negative_evals = True if next(neg_pos_gen)=='neg' else False
    spars = next(sparse_gen)
    matrix = None
    ev_range = [0,0]
    
    ev_range = np.random.randint(1,10,size=2)*np.random.random(size=2)
    if negative_evals:
        ev_range *= np.random.choice([-1,1],size=2)
    
    ev_range = np.sort(ev_range)
    print(ev_range)
    if mat_t == 'diagonal':
        sparsity = 1
        matrix = np.zeros((N,N))
        for idx in range(N):
            if (ev_range[1]-ev_range[0])!=0:
                matrix[idx,idx] = np.random.random()*(ev_range[1]-ev_range[0])+ev_range[0]
            else:
                matrix[idx,idx] = ev_range[0]
    elif mat_t=='hermitian':
        matrix = random_hermitian(N,eigrange=ev_range.tolist(),sparsity=spars)
    return matrix,mat_t,ev_range,spars,negative_evals

def transform_result(sv,negative_evals):
    res_list = []
    for d in sv.keys():
                    if negative_evals:
                        num = sum([2 ** -(i + 2)
                            for i, e in enumerate(reversed(d.split()[-2][:-1])) if e == "1"])
                        if d.split()[-2][-1] == '1':
                            num *= -1
                            if '1' not in d.split()[-2][:-1]:
                                num = -0.5
                    else:
                        num = sum([2 ** -(i + 1)
                            for i, e in enumerate(reversed(d.split()[-2])) if e == "1"])
                    if d.split()[0] == '1':
                        res_list.append(("Anc 1",num, np.complex(sv[d][0],sv[d][1]),d))
                    else:
                        res_list.append(("Anc 0",num, np.complex(sv[d][0],sv[d][1]),d))
    return res_list

def QPE_norm(y,w,k,n,t):
    #if only 1 eigenvalue present
    if all([w[i]==w[0] for i in range(len(w))]):
        return len(w)
    r = np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y))) /
                     (1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))
    r[np.isnan(r)]=2**k
    r = 2**(-2*k)*r**2
    return sum(r)
    

def is_resolvable(num,evo_time,num_bits):
    scale_num = evo_time*num/2/np.pi*2**num_bits
    if int(scale_num)!=scale_num:
           return False
    else:
           return True
           
def QPE_theory_ket(y,w,k,n,t,vec,neg_evals=True):
    #output: vector with ampl. of ket for each vector entry
    def nearest_el(ar,val):
        return np.argmin(np.abs(ar-val))
    resolvable = is_resolvable(w,t,k)
    x = None
    if neg_evals:
        y_ = np.arange(-2**(k-1),2**(k-1),1)
    else:
        y_ = y
    x_ =y_/t*2.0*np.pi/2.0**k

    if not resolvable:
        qpe_fac = 1/2**(k)*( (1-np.exp(1j*(2**k*w*t-2*np.pi*y))) /
                            (1-np.exp(1j*(w*t-2*np.pi*y/2**k))) )
    else:
        qpe_fac = np.array([0]*len(y))
        idx = nearest_el(x_,w)
        qpe_fac[idx] = 1
    outvec = np.zeros((len(y),len(vec))).astype(complex)
    
    
    for i,entr in enumerate(vec.tolist()):
        outvec[:,i] =qpe_fac*entr
        if x is None:
            x = x_
        else:
            x = np.dstack((x,x_))
    if neg_evals and not resolvable:
        c = int(len(y)/2)
        _  = outvec[c:,:].copy()
        outvec[c:,:] = outvec[:c,:]
        outvec[:c,:] = _
    
    return x[0,:,:],outvec

def superpose_QPE_kets(x_in,w_ar,k,n,t,vec_ar,neg_evals=True):
    '''assume equal superposition of vec'''
    output = np.zeros((len(x_in),len(vec_ar[0,:]))).astype(complex)
    norm = np.sqrt(QPE_norm(x_in,w_ar,k,n,t))
    for idx in range(len(w_ar)):
        w_i = w_ar[idx]
        v_i = vec_ar[:,idx]
        x,res = QPE_theory_ket(x_in,w_i,k,n,t,v_i,neg_evals)

        output += res
    output /= norm
    return (x.flatten(),output.flatten())


def test_with_QPE(config):

    backend = 'local_qasm_simulator'
    mode = config['mode']
    
    matrix_config = config.get('matrix')
    qpe_params =config.get('qpe')
    init_state_params = config.get("initial_state")
    reciprocal_params = config.get("reciprocal")


    if 'matrix' not in list(matrix_config.keys()):
        N = matrix_config['N']
        matrix,mat_t,ev_range,sparsity,negative_evals = matrix_generator(N)
    else:
        matrix = matrix_config['matrix']
        mat_t=''
        ev_range=[]
        sparsity=''
        negative_evals = matrix_config['negative_evals']
        N = matrix.shape[0]

    k = qpe_params['num_ancillae']
    w, v = np.linalg.eigh(matrix) 
    if any([w[i]<0 for i in range(N)]):
        negative_evals = True
        print("Negative EV present")
    else:
        negative_evals = False
        print("Only positive EV present")
        
    #set explicit pos / neg EV
    #negative_evals = True
    qpe_params['negative_evals'] = negative_evals
    reciprocal_params['negative_evals'] = negative_evals
    print("Matrix has Eigenvalues/vector:")
    print('\n\n'.join([str(w[i])+':'+'('+' '
                       .join(map(str,np.round(v[:,i],4).tolist()))+' )' for i in range(N)]))
    print("#"*20)
    if 'state_vector' not in list(init_state_params.keys()):
        invec_  = sum([v[:,i] for i in range(N)])
        invec = invec_ / np.sqrt(invec_.dot(invec_.conj()))
        #invec = v[:,0]
        init_state_params['state_vector']= invec
    print("Running test with input vector: {}".format(init_state_params['state_vector']))
    print("#"*20)
    init_state_params['num_qubits'] = int(np.ceil(np.log2(len(invec))))
    
    eigs = get_eigs_instance('QPE')
    eigs.init_params(qpe_params,matrix)
    init_state = get_initial_state_instance(init_state_params["name"])
    init_state.init_params(init_state_params)

    reci = get_reciprocal_instance(reciprocal_params["name"])
    reci.init_params(reciprocal_params)

    io_reg = QuantumRegister(init_state_params['num_qubits'],'io')
    qc = QuantumCircuit(io_reg)

    qc += init_state.construct_circuit("circuit",io_reg)
    qc += eigs.construct_circuit("circuit",io_reg)

    ev_reg = eigs._output_register
    evo_time = eigs._evo_time

    if mode == 'QPE_standalone': qc.snapshot("0")

    qc += reci.construct_circuit("circuit",ev_reg)

    if mode == 'QPE_ROT': qc.snapshot("1")

    data = execute(qc,backend=backend,shots=1,config={
        "data":["hide_statevector","quantum_state_ket"]}).result()
    if mode =='QPE_ROT':
        res = transform_result(
            data.get_data()["snapshots"]["1"]["quantum_state_ket"][0],negative_evals)
        anc_1 = [(d[1],d[2]) for d in res if d[0]=='Anc 1']
        x_res = []
        y_res = []
        for i in anc_1:
            x_res.append(i[0]/evo_time*2*np.pi)   
            y_res.append(np.abs(i[1]))
        x_n = np.arange(0,2**k)
        x_theo,y_theo = superpose_QPE_kets(x_n,w,k,0,evo_time,v,negative_evals)
        for idx in range(len(x_theo)):
            y_theo[idx] *= 2**-k/x_theo[idx]/evo_time*2*np.pi
        x_theo = x_theo[y_theo!=0]
        y_theo = y_theo[y_theo!=0]
        plt.close()
        plt.scatter(x_res,y_res,label='result',s=50,c='r')
        plt.scatter(x_theo,np.abs(y_theo),label='theory',s=30,c=(0,0,1,0.7))
        plt.ylabel('abs. of ket amplitude')
        plt.legend(loc='best')
        plt.title("Eigenvalues {}, EV range {}, type:{}".format(w,ev_range,mat_t))
        plt.show()

        return

def run_rotation_test(reci_type):
    k = 6
    
    qpe_params = {
        'name': 'QPE',
        'num_ancillae': k,
        'num_time_slices': 30,
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'hermitian_matrix': True,
        'paulis_grouping': 'random',
        "iqft": {
            "name": "STANDARD"
        },
    }

    
    reci_params = {
        "name": reci_type,
        }
    
    if reci_type=='LOOKUP':
        n = min(k-1,5)
        reci_params['pat_length'] = n
        reci_params['subpat_length']=int(np.ceil(n/2))
    elif reci_type=='GENCIRCUITS':
        reci_params['scale'] = 2

    initial_state_params = {
        'name':'CUSTOM',
    }

    matrix_params = {
                'N':2,             
        }
    config = {'mode':'QPE_ROT',
              'initial_state':initial_state_params,
              'reciprocal':reci_params,
              'qpe':qpe_params,
              'matrix':matrix_params,
            }
    
   
    test_with_QPE(config)
np.random.seed(0)
if __name__=='__main__':
    for i in range(10):
        run_rotation_test('GENCIRCUITS')

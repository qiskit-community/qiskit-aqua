from qiskit_aqua import get_reciprocal_instance, get_initial_state_instance
from qiskit.extensions.simulator import snapshot
from qiskit import QuantumCircuit,QuantumRegister
from qiskit import execute
from qiskit.tools.visualization import circuit_drawer
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

def transform_result(sv,negative_evals):
    res_list = []
    for d in sv.keys():
                    if negative_evals:
                        num = sum([2 ** -(i + 2)
                            for i, e in enumerate(reversed(d.split()[-1][:-1])) if e == "1"])
                        if d.split()[-1][-1] == '1':
                            num *= -1
                            if '1' not in d.split()[-1][:-1]:
                                num = -0.5
                    else:
                        num = sum([2 ** -(i + 1)
                            for i, e in enumerate(reversed(d.split()[-1])) if e == "1"])
                    if d.split()[0] == '1':
                        res_list.append(("Anc 1",num, np.complex(sv[d][0],sv[d][1]),d))
                    else:
                        res_list.append(("Anc 0",num, np.complex(sv[d][0],sv[d][1]),d))
    return res_list

def test_value_range(k,reci_type='LOOKUP'):
        backend = 'local_qasm_simulator'
        negative_evals = False
        params = {
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "CUSTOM",
                "state_vector": []
            },
            "reciprocal": {
                "name": reci_type,
                'negative_evals':negative_evals,
               
            }
        }
        if reci_type=='LOOKUP':
            n = min(k-1,5)
            params['reciprocal']['pat_length'] = n
            params['reciprocal']['subpat_length']=int(np.ceil(n/2))
        elif reci_type=='GENCIRCUITS':
            params['reciprocal']['scale'] = 2
        x = []
        y = []
            
        for pattern in itertools.product('01', repeat=k):
            if not '1' in pattern:
                continue
            state_vector = get_statevector_representation(list(pattern))
            if negative_evals:
                num = np.sum([2 ** -(n + 2)
                              for n, i in enumerate(reversed(pattern[:-1])) if i == '1'])
                if pattern[-1] == '1':
                    num *= -1
                if pattern[-1] and not '1' in pattern[:-1]:
                    num = -0.5
            else:
                num = np.sum([2 ** -(n + 1)
                              for n, i in enumerate(reversed(pattern)) if i == '1'])

            params["initial_state"]["state_vector"] = state_vector
            params["initial_state"]["num_qubits"] = k
            print(state_vector)
            print("Next pattern:", pattern, "Numeric:",num)
            init_state_params = params.get("initial_state")
            init_state = get_initial_state_instance(init_state_params["name"])
            init_state.init_params(init_state_params)
            reciprocal_params = params.get("reciprocal")
            #print(reciprocal_params)
            reci = get_reciprocal_instance(reciprocal_params["name"])
            reci.init_params(reciprocal_params)
            inreg = QuantumRegister(k,'io')
            qc = QuantumCircuit(inreg)
            qc += init_state.construct_circuit("circuit",inreg)
            st_len = qc.number_atomic_gates()
            qc += reci.construct_circuit("circuit",inreg)
            tot_len = qc.number_atomic_gates()-st_len
            print("Length of rotation circuit:",tot_len)
            print(set([str(type(gate)) for gate in qc.data]))
            qc.snapshot("0")
            
            data = execute(qc,backend=backend,shots=1,config={
                                 "data":["hide_statevector","quantum_state_ket"]}).result()
            res = transform_result(
                data.get_data()["snapshots"]["0"]["quantum_state_ket"][0],negative_evals)
            anc_1 = [(d[1],d[2]) for d in res if d[0]=='Anc 1']
            for i in anc_1:
                x.append(i[0]) 
                y.append(i[1]*2**k)
        if negative_evals:
            x_ = np.append(np.linspace(-0.5,-2**-k,500),np.linspace(2**-k,0.5,500))
        else:
            x_ = np.linspace(0,1,1000)

        plt.scatter(x,y)
        plt.plot(x_[:int(len(x_)/2)],1/x_[:int(len(x_)/2)],c='r')
        plt.plot(x_[int(len(x_)/2):],1/x_[int(len(x_)/2):],c='r')
        plt.title("Circuit length: {}".format(tot_len))
        
        plt.show()
        return

if __name__=='__main__':
    k=8
    test_value_range(k,'LOOKUP')

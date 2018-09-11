import numpy as np
import itertools

from hybrid_rotation import hybrid_rot
import matplotlib.pyplot as plt
def get_statevector_representation(bitpattern):
    '''Using the tensorproduct of the qubits representing the bitpattern, estimate
    the input state vector to the rotation routine

    Args:
        bitpattern (list)'''
    state_vector = None
    for bit in bitpattern:
        vec = np.array([0,1]) if bit =='1' else np.array([1,0])
        if state_vector is None:
            state_vector = vec
            continue
        state_vector = np.tensordot(state_vector,vec,axes=0)
    print(state_vector.flatten().shape)
    return (state_vector.flatten())

def test_valuerange(k,n):
    res_dict = {}
    for pattern in itertools.product('01',repeat=k):
        state_vector = get_statevector_representation(list(pattern))
        draw = False
        #state_vector = np.zeros(2**k)
        #state_vector[4] = 1
        initial_params = {
            "name": "CUSTOM",
            "num_qubits" : k,
            "state_vector": state_vector if not draw else []
        }
        print(state_vector)
        obj= hybrid_rot(k,n,initial_params,measure=True)
        res = (obj.set_up_and_execute_circuit(k,n))
        
        for d in res:
            if d[1][0] == '1':
                res_dict.update({float(d[2].split()[-1]):d[0]})
    vals = list(res_dict.keys())
    inverse = [res_dict[_] for _ in vals]
    vals = np.array(vals)
    inverse = np.array(inverse)
    plt.scatter(vals,inverse/2**(-len(res[0][1].split()[-1])))
    plt.plot(np.linspace(np.min(vals),np.max(vals),1000),1/np.linspace(np.min(vals),np.max(vals),1000))
    plt.show()

test_valuerange(5,2)
#state_vector = get_statevector_representation(['1','1','0','0','1','0'])#,'1','0'])

#k = 6
#n = 4
#draw = False
#state_vector = np.zeros(2**k)
#state_vector[4] = 1
#initial_params = {
#            "name": "CUSTOM",
#            "num_qubits" : k,
#            "state_vector": state_vector if not draw else []
#}
#print(state_vector)
#obj= hybrid_rot(k,n,initial_params,measure=True)
#print(obj.set_up_and_execute_circuit(k,n))

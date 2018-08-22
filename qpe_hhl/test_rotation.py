import numpy as np
from math import log
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, register, execute#, ccx
from qiskit import register, available_backends, get_backend, least_busy
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance

from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer, plot_histogram
import matplotlib.pyplot as plt
import itertools

from rotation import C_ROT

results_x = []
results_y = []
qbits = 5
nege = True
for i in range(2**qbits):
    state_vector = np.zeros(2**qbits)
    # state_vector[15] = 1
    state_vector[i] = 1

    params = {
        'algorithm': {
                'name': 'C_ROT',
                'num_qbits_precision_ev': qbits,
                'negative_evals': nege,
                'previous_circuit': None,
                'backend' : "local_qasm_simulator",
                #'evo_time': 2*np.pi/4,
                #'use_basis_gates': False,
        },
        #for running the rotation seperately, supply input
        "initial_state": {
            "name": "CUSTOM",
            "state_vector": state_vector
        }
    }

    crot = C_ROT()
    crot.init_params(params)
    #qc = crot._setup_crot(measure=True)
    #drawer(qc, filename="firstnewton.png")

    results = crot.run()['measurements']
    print(results)
    results_x.append(float(results[0][2].split()[-1]))
    results_y.append(float(results[0][2].split()[0]))
    #drawer(qc, filename="firstnewton.png")
    #plt.show()


def theory(qbits,sign_flag =False):
    x = []
    y = []
    for n,i in enumerate(map(''.join, itertools.product('01',repeat=qbits))):
        if sign_flag:
            if i[-1]=='1':
                x.append(sum([2 ** (m) for m, e in enumerate((i[:-1])) if e ==
                               "1"]))
                print(''.join([_ for _ in reversed(i)]))
                y_ex = ''.join([_ for _ in reversed(i)])
                y_ = y_ex.find('1')
                print(y_,type(y_ex))
                if y_ != -1:
                    y.append(2**y_)
                else:
                    y.append(0)
        else:

            x.append(sum([2 ** (m) for m, e in enumerate((i)) if e ==
                           "1"]))

            y_ex = ''.join([_ for _ in reversed(i)]).find('1')
            if y_ex != -1:
                y.append(2**y_ex)
            else:
                y.append(0)
    return x,y
x,y = theory(qbits,nege)
plt.scatter(results_x,results_y)
plt.scatter(x,y)
dx = np.linspace(np.min(results_x),np.max(results_x),1000)
plt.plot(dx,1/dx)
plt.ylim([-32,32])
plt.show()

import numpy as np
from qiskit_aqua.hhl import QPE
from qiskit import execute, ClassicalRegister, QuantumRegister
#from qiskit.tools.visualization import matplotlib_qcuit_drawer
from copy import deepcopy

matrix = np.array([[0.49997724, 0.2491572 ], [0.2491572,  0.50002276]])
#matrix = np.array([[1, -1j], [1j,  1]])

qpe = QPE()
n = 2
params = {
        "algorithm": {
            "num_ancillae": n,
            "num_time_slices": 50
            },
        "initial_state": {
            "name": "CUSTOM",
            "state_vector": [1/2**0.5, 1/2**0.5]
            }
        }


qpe.init_params(params, matrix)

qc = qpe._setup_qpe()
a = qc.regs["a"]
x = QuantumRegister(1)
qc.add(x)

qc.cu3(np.pi/4, 0, 0, a[1], x[0])
qc.cu3(np.pi/2, 0, 0, a[0], x[0])

qc += qpe._construct_inverse()


qcr = ClassicalRegister(1)
qc.add(qcr)
qc.measure(x, qcr)

res = execute(qc, "local_qasm_simulator").result()
print(res.get_counts())

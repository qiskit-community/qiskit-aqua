import numpy as np
from qiskit_aqua.hhl import QPE
from qiskit import execute, ClassicalRegister

matrix = np.array([[0.49997724, 0.2491572 ], [0.2491572,  0.50002276]])
#matrix = np.array([[1, 0], [0,  2]])

qpe = QPE()

params = {
        "algorithm": {
            "num_ancillae": 4,
            "num_time_slices": 100
            },
        "initial_state": {
            "name": "CUSTOM",
            "state_vector": [0, 1]
            }
        }


qpe.init_params(params, matrix)

circ = qpe._setup_qpe()
circ += qpe._construct_inverse()

cr = ClassicalRegister(4)
qcr = ClassicalRegister(1)
circ.add(cr, qcr)
circ.measure(circ.get_qregs()["a"], cr)
circ.measure(circ.get_qregs()["q"], qcr)


res = execute(circ, "local_qasm_simulator").result()
print(res.get_counts())

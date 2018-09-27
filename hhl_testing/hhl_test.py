# from qiskit import QuantumRegister, QuantumCircuit, execute
# import qiskit.extensions.simulator

from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance
from qiskit_aqua.utils import random_hermitian, random_non_hermitian

import numpy as np
import pickle

import matplotlib.pyplot as plt
import copy

# from qiskit_aqua.algorithms.components.reciprocals.lookup_rotation import LookupRotation
# from qiskit_aqua.algorithms.single_sample.hhl.hhl import HHL

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 1,
        "expansion_mode": "trotter",
        "negative_evals": False,
        "num_ancillae": 6,
    },
    "reciprocal": {
        "name": "LOOKUP",
        "lambda_min": 1,
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    }
}

matrix = random_hermitian(2, K=(3, 1, 1))

result = []

N, M = 2, 40

# for j in range(N):
#     res = []
#     invec = np.random.random(2)
#     d = {"input": {"matrix": matrix, "invec": invec, "params": params}}
#     for i in range(50, 10+10*M, 10):
#         params["eigs"]["num_time_slices"] = i
#         ret = run_algorithm(params, (matrix, invec))
#         # ret = {"fidelity": 1}
#         d["output"] = ret
#         res.append(copy.deepcopy(d))
#     result.append(res)
# pickle.dump(result, open("result.pkl", "wb"))
result = pickle.load(open("result.pkl", "rb"))

matrix = result[0][0]["input"]["matrix"]
w, v = np.linalg.eig(matrix)

print(w)


xs = []
k = []
for res in result:
    x = np.array(list(map(lambda x: [x["output"]["fidelity"],
        x["input"]["params"]["eigs"]["num_time_slices"]], res)))
    plt.plot(x[:, 1], x[:, 0])
    xs += [x[:, 0]]
    k.append(x[:, 1])
# plt.plot(k[0], abs(xs[0]-xs[1]))
plt.show()
    

# print(matrix.dot(ret["solution"]))


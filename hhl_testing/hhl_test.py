from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance
from qiskit_aqua.utils import random_hermitian, random_non_hermitian

import numpy as np
import pickle

import matplotlib.pyplot as plt
import copy

import json
import hashlib
import os

def clean_dict(params, matrix, invec):
    decomplex = lambda x: [x.real, x.imag] if isinstance(x, (complex, np.complex128)) else x
    params["matrix"] = list(map(lambda y: list(map(decomplex, y)), matrix.tolist()))
    params["invec"] = list(map(decomplex, invec.tolist()))

def get_hash(params, matrix, invec):
    clean_dict(params, matrix, invec)
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def register_hash(params, matrix, invec):
    ha = get_hash(params, matrix, invec)
    with open("data/index.json", "r") as f:
        d = json.load(f)
    if not ha in d:
        d[ha] = params
        with open("data/index.json", "w") as f:
            json.dump(d, f, sort_keys=True, indent=2)
        return ha
    return None

def cleanup_index():
    r = {}
    with open("data/index.json", "r") as f:
        d = json.load(f)
        for key in d.keys():
            if key in os.listdir("data/"):
                r[key] = d[key]
    with open("data/index.json", "w") as f:
        json.dump(r, f, sort_keys=True, indent=2)

def save_data(params, matrix, invec, data):
    ha = register_hash(params, matrix, invec)
    if ha:
        with open(os.path.join("data", ha), "wb") as f:
            pickle.dump(data, f)
    return ha

def load_data(params, matrix, invec):
    ha = get_hash(params, matrix, invec)
    if ha in os.listdir("data/"):
        with open(os.path.join("data", ha), "rb") as f:
            data = pickle.load(f)
            return data

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

problem_size = [2, 4]
negative_evals = [False, True]

expansion_mode = ["trotter", "suzuki"]

num_time_slices = {}
num_time_slices["trotter"] = np.arange(10, 400, 20)
num_time_slices["suzuki"] = np.arange(10, 100, 10)

expansion_order = {}
expansion_order["trotter"] = [1]
expansion_order["suziki"] = [1, 2, 3]

reciprocals = ["LOOKUP"]







matrix = random_hermitian(2, K=(3, 1, 1))
save_data(params, matrix, invec, np.array([1, 2]))
res = load_data(params, matrix, invec)

cleanup_index()

#
# result = []
#
# N, M = 2, 40
#
# # for j in range(N):
# #     res = []
# #     invec = np.random.random(2)
# #     d = {"input": {"matrix": matrix, "invec": invec, "params": params}}
# #     for i in range(50, 10+10*M, 10):
# #         params["eigs"]["num_time_slices"] = i
# #         ret = run_algorithm(params, (matrix, invec))
# #         # ret = {"fidelity": 1}
# #         d["output"] = ret
# #         res.append(copy.deepcopy(d))
# #     result.append(res)
# # pickle.dump(result, open("result.pkl", "wb"))
# result = pickle.load(open("result.pkl", "rb"))
#
# matrix = result[0][0]["input"]["matrix"]
# w, v = np.linalg.eig(matrix)
#
# print(w)
#
#
# xs = []
# k = []
# for res in result:
#     x = np.array(list(map(lambda x: [x["output"]["fidelity"],
#         x["input"]["params"]["eigs"]["num_time_slices"]], res)))
#     plt.plot(x[:, 1], x[:, 0])
#     xs += [x[:, 0]]
#     k.append(x[:, 1])
# # plt.plot(k[0], abs(xs[0]-xs[1]))
# plt.show()
#     
#
# # print(matrix.dot(ret["solution"]))


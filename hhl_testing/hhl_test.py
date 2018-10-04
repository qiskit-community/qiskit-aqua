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
import time
import itertools

test_object_matrices = None
test_object_vectors = None

def clean_dict(params, matrix, invec):
    params["eigs"]["num_ancillae"] = int(params["eigs"]["num_ancillae"])
    params["eigs"]["num_time_slices"] = int(params["eigs"]["num_time_slices"])
    decomplex = lambda x: [x.real, x.imag] if isinstance(x, (complex, np.complex128)) else x
    params["matrix"] = list(map(lambda y: list(map(decomplex, y)), matrix.tolist()))
    params["invec"] = list(map(decomplex, invec.tolist()))

def get_hash(params, matrix, invec):
    params = copy.deepcopy(params)
    clean_dict(params, matrix, invec)
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def register_hash(params, matrix, invec, force=False):
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/index.json"):
        os.system('echo "{}" > data/index.json')
    ha = get_hash(params, matrix, invec)
    with open("data/index.json", "r") as f:
        d = json.load(f)
    if (not ha in d) or force:
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

def save_data(params, matrix, invec, data, force=False):
    ha = register_hash(params, matrix, invec, force)
    if ha:
        with open(os.path.join("data", ha), "wb") as f:
            pickle.dump(data, f)
    return data

def load_data(params, matrix, invec):
    ha = get_hash(params, matrix, invec)
    try:
        if ha in os.listdir("data/"):
            with open(os.path.join("data", ha), "rb") as f:
                data = pickle.load(f)
                return data
    except FileNotFoundError:
        pass

def get_matrix(index, n=2, name="eigrange=(1,10)"):
    if index < 0:
        name = "K=(3,1)"
        index = -index-1
    global test_object_matrices
    if test_object_matrices is None:
        test_object_matrices = pickle.load(
                open("test_objects/matrices_100_n=%d_%s.pkl" % (n, name), 
                    "rb"))
    return test_object_matrices[index]

def get_vector(index, n=2):
    global test_object_vectors
    if test_object_vectors is None:
        test_object_vectors = pickle.load(
                open("test_objects/vectors_100_n=%d.pkl" % n, "rb"))
    return test_object_vectors[index]


def run_test(params, force=False):
    matrix = params["input"]["matrix"]
    vector = params["input"]["vector"]
    del params["input"]
    if isinstance(matrix, int):
        matrix = get_matrix(matrix)
    if isinstance(vector, int):
        vector = get_vector(vector)
    data = load_data(params, matrix, vector)
    if (not data) or force:
        t = time.time()
        data = run_algorithm(params, (matrix, vector))
        data["time_elapsed"] = time.time() - t
        save_data(params, matrix, vector, data, force)
    return data


def run_tests(params, force=False, status=None):
    tests = {}
    for module, param in params.items():
        for key, value in param.items():
            if isinstance(value, (list, tuple)):
                tests[module + " " + key] = value
    keys = list(tests.keys())
    vals = list(map(lambda x: tests[x], keys))
    l = 1
    for val in vals:
        l *= len(val)
    ret = {"keys": keys, "values": [], "data": []}
    c = 0
    for items in itertools.product(*vals):
        c += 1
        print("%d/%d" % (c, l), *list(zip(keys, items)))
        ret["values"].append(items)
        p = copy.deepcopy(params)
        for key, val in zip(keys, items):
            k1, k2 = key.split(" ")
            p[k1][k2] = val
        ret["data"].append(run_test(p, force))
        if status:
            status(ret["data"][-1])
        #ret["data"].append(p)
    return ret


def get_for(params, data, subs=None):
    ret = {"vals": [], "data": []}
    idxs = list(map(lambda x: data["keys"].index(x), params.keys()))
    vals = list(map(lambda x: params[x], params.keys()))
    ret["keys"] = [key for key in data["keys"] if key not in params]
    for v, res in zip(data["values"], data["data"]):
        cont = True
        for x, y in zip(vals, [v[i] for i in idxs]):
            if x != y:
                cont = False
                break
        if cont:
            ret["vals"].append([v[i] for i in range(len(v)) if i not in idxs])
            if subs:
                res = res[subs]
            ret["data"].append(res)
    if len(ret["keys"]) == 1:
        return ret["data"]
    return ret
        
if __name__ == "__main__":
    params = {
        "algorithm": {
            "name": "HHL",
            "mode": "state_tomography"
        },
        "eigs": {
            "name": "QPE",
            "num_time_slices": list(range(10, 110, 20)),
            "expansion_mode": "suzuki",
            "expansion_order": 2,
            "negative_evals": False,
            "num_ancillae": (4, 6, 8),
        },
        "reciprocal": {
            "name": "LOOKUP",
            "lambda_min": 1,
        },
        "backend": {
            "name": "local_qasm_simulator",
            "shots": 1
        },
        "input": {
            "matrix": 4,
            "vector": 7,
        }
    }
    
    sol = np.linalg.solve(get_matrix(params["input"]["matrix"]),
            get_vector(params["input"]["vector"]))

    print(sol, sol/np.linalg.norm(sol))

    res = run_tests(params, status=lambda res: res["fidelity"])
    data = [get_for({"eigs num_ancillae": i}, res, "fidelity") for i in params["eigs"]["num_ancillae"]]

    plt.imshow(data, cmap='hot')
    plt.show()
    
    for i in range(len(data)):
        plt.plot(params["eigs"]["num_time_slices"], data[i])
    plt.show()

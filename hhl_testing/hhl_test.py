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

test_object_matrices = None
test_object_vectors = None

def clean_dict(params, matrix, invec):
    params["eigs"]["num_ancillae"] = int(params["eigs"]["num_ancillae"])
    params["eigs"]["num_time_slices"] = int(params["eigs"]["num_time_slices"])
    decomplex = lambda x: [x.real, x.imag] if isinstance(x, (complex, np.complex128)) else x
    params["matrix"] = list(map(lambda y: list(map(decomplex, y)), matrix.tolist()))
    params["invec"] = list(map(decomplex, invec.tolist()))

def get_hash(params, matrix, invec):
    clean_dict(params, matrix, invec)
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def register_hash(params, matrix, invec):
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/index.json"):
        os.system('echo "{}" > data/index.json')
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


# reciprocal, problem_size, negative_evals, num_ancillae, expansion_mode,
# expansion_order, num_time_slices, matrix_type, vector, output
def run_test(matrix, invec, reciprocal="LOOKUP", n=2, negative_evals=False, 
        num_ancillae=6, expansion_mode="trotter", expansion_order=1,
        num_time_slices=10, time_data=None):
    params = {
        "algorithm": {
            "name": "HHL",
            "mode": "state_tomography"
        },
        "eigs": {
            "name": "QPE",
            "num_time_slices": num_time_slices,
            "expansion_mode": expansion_mode,
            "expansion_order": expansion_order,
            "negative_evals": negative_evals,
            "num_ancillae": num_ancillae,
        },
        "reciprocal": {
            "name": reciprocal,
            "lambda_min": 1,
        },
        "backend": {
            "name": "local_qasm_simulator",
            "shots": 1
        }
    }
    data = load_data(params.copy(), matrix, invec)
    if not data:
        t = time.time()
        res = run_algorithm(params, (matrix, invec))
        res["time_elapsed"] = time.time() - t
        if time_data is not None:
            time_data.append(res["time_elapsed"])
        data = save_data(params, matrix, invec, res)
    return data


def run_all(limit_inputs=5):
    input_files = os.listdir("test_objects")
    input_dict = {}
    for name in input_files:
        with open(os.path.join("test_objects", name), "rb") as f:
            input_dict[name[:-4]] = pickle.load(f)

    vectors = {2: input_dict["vectors_100_n=2"][:limit_inputs],
            4: input_dict["vectors_100_n=4"][:limit_inputs]}

    matrices = {
        2: {
    #        "fixed_condition": input_dict["matrices_100_n=2_K=(3,1)"][:limit_inputs],
            "eigrange": input_dict["matrices_100_n=2_eigrange=(1,10)"][:limit_inputs],
            "negative_evals": input_dict["matrices_100_n=2_eigrange=(-10,10)"][:limit_inputs],
        },
        4: {
    #        "fixed_condition": input_dict["matrices_100_n=4_K=(3,1)"][:limit_inputs],
            "eigrange": input_dict["matrices_100_n=4_eigrange=(1,10)"][:limit_inputs],
            "negative_evals": input_dict["matrices_100_n=4_eigrange=(-10,10)"][:limit_inputs],
        }
    }

    problem_size = [2, 4]
    negative_evals = [False, True]

    num_ancillae = np.arange(4, 10)

    expansion_mode = ["trotter", "suzuki"]

    num_time_slices = {}
    num_time_slices["trotter"] = np.arange(10, 400, 20)
    num_time_slices["suzuki"] = np.arange(10, 100, 10)

    expansion_order = {}
    expansion_order["trotter"] = [1]
    expansion_order["suzuki"] = [1, 2, 3]

    reciprocals = ["LOOKUP"]

    total_simulatiations = (len(reciprocals) * len(problem_size)
        * len(num_ancillae) * (limit_inputs**2) * (len(expansion_order["trotter"])
            * len(num_time_slices["trotter"]) + len(expansion_order["suzuki"])
            * len(num_time_slices["suzuki"])) * 2)
    c = 1
    time_data = []
    for r in reciprocals:
        for n in problem_size:
            for ne in negative_evals:
                for na in num_ancillae:
                    for em in expansion_mode:
                        for eo in expansion_order[em]:
                            for nts in num_time_slices[em]:
                                for mtype, mats in matrices[n].items():
                                    if ne == (mtype == "negative_evals"):
                                        for matrix in mats:
                                            for invec in vectors[n]:
                                                run_test(matrix, invec, 
                                                        r, n, ne, na, em, eo, nts,
                                                        time_data=time_data)
                                                print("%d/%d"%(c,
                                                    total_simulatiations))
                                                if len(time_data) > 0:
                                                    avg = sum(time_data)/len(time_data)
                                                    est = avg*(total_simulatiations-c)
                                                    print("took %.3fs | avg %.3fs"
                                                            "| estimated %.3fs"
                                                            % (time_data[-1], avg, est))
                                                c += 1


run_all(5)

# matrix = get_matrix(0)
# invec = get_vector(0)
#
# num_time_slices = np.arange(10, 50, 10)
#
# y = []
#
# c = 0
# for nts in num_time_slices:
#     c += 1
#     print("%d/%d"%(c, len(num_time_slices)))
#     y.append(run_test(matrix, invec, expansion_mode="suzuki", expansion_order="2",
#         num_time_slices=nts)["fidelity"])
#
# plt.plot(num_time_slices, y)
# plt.show()


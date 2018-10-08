from qiskit_aqua.utils import random_hermitian, random_non_hermitian
import numpy as np
import pickle, json, os, copy

TEST_BASE_DIR = "test_objects"

test_objects = {}

default_params = {
    "type": "generate",
    "test_set": "specified",
    "condition": 2,
    "n": 2,
    "lambda_min": 1,
    "repetition": 0,
    "vector": np.array([1, 1]),
    "hermitian": True,
    "negative_evals": True
}

def jsonify(params, dels=None):
    p = copy.deepcopy(params["input"])
    if dels:
        for d in dels:
            del p[d]
    if "vector" in p and isinstance(p["vector"], np.ndarray):
        p["vector"] = p["vector"].tolist()
    return json.dumps(p, sort_keys=True)

def fillup(params):
    for key in default_params:
        if not key in params["input"]:
            if key in ["negative_evals", "hermitian"] and key in params["eigs"]:
                params["input"][key] = params["eigs"][key]
            else:
                params["input"][key] = default_params[key]

def generate_matrix(params):
    p = params["input"]
    matrix = random_hermitian if p["hermitian"] else random_non_hermitian
    mat = matrix(p["n"], K=(p["condition"], p["lambda_min"],
            -1 if p["negative_evals"] else 1))
    return mat

def generate_vector(params, mat):
    p = params["input"]
    if p["hermitian"]:
        w, v = np.linalg.eigh(mat)
        vec = v.dot(p["vector"])
    else:
        NotImplementedError()
    return vec

def generate_input(params):
    fillup(params)
    test_set = params["input"]["test_set"]

    global test_objects
    if test_set not in test_objects:
        if os.path.exists(os.path.join(TEST_BASE_DIR, test_set + ".pkl")):
            with open(os.path.join(TEST_BASE_DIR, test_set + ".pkl"), "rb") as f:
                test_objects[test_set] = pickle.load(f)
        else:
            test_objects[test_set] = {}
    
    vec_key = jsonify(params)
    mat_key = jsonify(params, dels=["vector"])
    if not mat_key in test_objects[test_set]:
        mat = generate_matrix(params)
        test_objects[test_set][mat_key] = mat
    else:
        mat = test_objects[test_set][mat_key]
    if not vec_key in test_objects[test_set]:
        vec = generate_vector(params, mat)
        test_objects[test_set][vec_key] = vec
    else:
        vec = test_objects[test_set][vec_key]

    return {"n": params["input"]["n"], "matrix": mat, "vector": vec}

def save_generated_inputs():
    for k, v in test_objects.items():
        with open(os.path.join(TEST_BASE_DIR, k+".pkl"), "wb") as f:
            pickle.dump(v, f)


#
#
# cond = (2, 3, 5, 8, 10, 30, 50, 100)
#
# data = {}
#
# for ne in [True, False]:
#     data[ne] = {}
#     for n in [2, 4]:
#         data[ne][n] = []
#         for c in cond:
#             v = []
#             h = random_hermitian(n, K=(c, 1, -1 if ne else 1))
#             data[ne][n].append((h, []))
#             w = np.linalg.eigh(h)[1]
#             v.append(w.dot(np.ones(n)))
#             if n == 2:
#                 v.append(w.dot(np.array([0.8, 0.2])))
#                 v.append(w.dot(np.array([0.2, 0.8])))
#             for vi in v:
#                 data[ne][n][-1][1].append(vi)
# print(data)
# with open("test_objects/specified.pkl", "wb") as f:
#     pickle.dump(data, f)

if __name__ == "__main__":

    params = {
        "algorithm": {
            "name": "HHL",
            "mode": "state_tomography"
        },
        "eigs": {
            "name": "QPE",
            "num_time_slices": 4,
            "expansion_mode": "suzuki",
            "expansion_order": 2,
            "negative_evals": False,
            "num_ancillae": 4,
        },
        "reciprocal": {
            "name": "LOOKUP",
            "lambda_min": 0.9,
            "pat_length": 4
        },
        "backend": {
            "name": "local_qasm_simulator",
            "shots": 1
        },
        "input": {
            "type": "generate",
            "n": 2,
            "test_set": "specified",
            "condition": 20,
            "repetition": 1,
            "vector": np.array([1, 3])
        }
    }

    print(generate_input(params))
    save_generated_inputs()

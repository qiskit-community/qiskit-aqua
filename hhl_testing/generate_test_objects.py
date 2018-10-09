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
    "hermitian": True,
    "negative_evals": True
}

# Create TEST_BASE_DIR if not exists
if not os.path.exists(TEST_BASE_DIR):
    os.mkdir(TEST_BASE_DIR)


def jsonify(params, dels=None):
    """
    Truns the params dict into a json string.
    Refactor vector into python list if neccessary.
    """
    p = copy.deepcopy(params["input"])
    if dels:
        for d in dels:
            del p[d]
    if "vector" in p and isinstance(p["vector"], np.ndarray):
        p["vector"] = p["vector"].astype(float).tolist()
    return json.dumps(p, sort_keys=True)


def fillup(params):
    """
    Merge the default parameters with the specified ones.
    """
    for key in default_params:
        if not key in params["input"]:
            if key in ["negative_evals", "hermitian"] and key in params["eigs"]:
                params["input"][key] = params["eigs"][key]
            else:
                params["input"][key] = default_params[key]
    # Create a equal eigenvector weight
    if not "vector" in params["input"]:
        params["input"]["vector"] = np.ones(params["input"]["n"])


def generate_matrix(params):
    """
    Generate a random matrix with condition specified in params
    """
    p = params["input"]
    matrix = random_hermitian if p["hermitian"] else random_non_hermitian
    mat = matrix(p["n"], K=(p["condition"], p["lambda_min"],
            -1 if p["negative_evals"] else 1))
    return mat

def generate_vector(params, mat):
    """
    Generate a vector as superposition of eigenvectors of the matrix mat.
    The weights of the superposition are specified in params.
    """
    p = params["input"]
    if p["hermitian"]:
        w, v = np.linalg.eigh(mat)
        vec = v.dot(p["vector"])
    else:
        NotImplementedError()
    return vec


def generate_input(params):
    """
    Generate input dict for params
    Args:
        params: dict:
            n (int): problem size (default: 2)
            type (str): 'generate' used as breakeout from hhl_test_suite
            test_set (str): some random test_set name (default: specified)
            condition (int): condition number of matrix (default: 2)
            
    """
    fillup(params)
    test_set = params["input"]["test_set"]

    # Check if test set is already loaded, otherwise create
    global test_objects
    if test_set not in test_objects:
        if os.path.exists(os.path.join(TEST_BASE_DIR, test_set + ".pkl")):
            with open(os.path.join(TEST_BASE_DIR, test_set + ".pkl"), "rb") as f:
                test_objects[test_set] = pickle.load(f)
        else:
            test_objects[test_set] = {}
    # Check if mat, vec with params is in test_set, otherwise generate
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
    """
    Save the test_set dict.
    """
    for k, v in test_objects.items():
        with open(os.path.join(TEST_BASE_DIR, k+".pkl"), "wb") as f:
            pickle.dump(v, f)

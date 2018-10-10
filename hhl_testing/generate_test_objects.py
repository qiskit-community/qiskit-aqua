from qiskit_aqua.utils import random_hermitian, random_non_hermitian
import numpy as np
import pickle, json, os, copy, hashlib

TEST_BASE_DIR = "test_objects"

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


def get_hash(s):
    """
    get hash of string
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jsonify(params, dels=None):
    """
    Truns the params dict into a json string.
    Refactor vector into python list if neccessary.
    """
    p = copy.deepcopy(params["input"])
    if dels:
        for d in dels:
            del p[d]
    for key in list(p.keys()):
        if (not key == "vector") and default_params[key] == p[key]:
            del p[key]
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


def get_matrix(params):
    """
    try to load matrix with specific params, generate and save if not exists
    """
    file_name = get_hash(jsonify(params, dels=["vector"]))
    path = os.path.join(TEST_BASE_DIR, file_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
    else:
        ret = generate_matrix(params)
        with open(path, "wb") as f:
            pickle.dump(ret, f)
    return ret


def get_vector(params, matrix):
    """
    try to load vector with specific params, generate and save if not exists
    """
    file_name = get_hash(jsonify(params))
    path = os.path.join(TEST_BASE_DIR, file_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
    else:
        ret = generate_vector(params, matrix)
        with open(path, "wb") as f:
            pickle.dump(ret, f)
    return ret


def generate_input(params):
    """
    Generate input dict for params
    Args:
        params["input"]: dict:
            n (int): problem size (default: 2)
            type (str): 'generate' used as breakeout from hhl_test_suite
            test_set (str): some random test_set name (default: specified)
            condition (int): condition number of matrix (default: 2)
            
    """
    # Merge defaults
    fillup(params)

    # Check if mat, vec with params is in test_set, otherwise generate
    mat = get_matrix(params)
    vec = get_vector(params, mat)

    return {"n": params["input"]["n"], "matrix": mat, "vector": vec}

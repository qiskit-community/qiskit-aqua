import numpy as np
from qiskit_aqua import run_algorithm

import os
import itertools
import copy
import pickle
import hashlib
import json

BASE_DIR = "data2"

RAW_DIR = "raw"
INFO_DIR = "info"

TEST_DATA = "test_objects"

def check_dirs():
    """ check if data directories exists and create if not so """
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    if not os.path.exists(os.path.join(BASE_DIR, INFO_DIR)):
        os.mkdir(os.path.join(BASE_DIR, INFO_DIR))
    if not os.path.exists(os.path.join(BASE_DIR, RAW_DIR)):
        os.mkdir(os.path.join(BASE_DIR, RAW_DIR))

def _gen_dict(keys, values):
    ret = {}
    for key, val in zip(keys, values):
        k1, k2 = key.split(" ")
        if k1 not in ret:
            ret[k1] = {}
        ret[k1][k2] = val
    return ret

def _tuple(obj):
    try:
        iter(obj)
        return tuple(obj)
    except TypeError:
        return (obj,)

def product(grid_params, cond_grid_params):
    pars = lambda x: _gen_dict(list(grid_params.keys())
            + list(cond_grid_params.keys()), x)
    result = [[]]
    for pool in grid_params.values():
        result = [x+[y] for x in result for y in _tuple(pool)]
    for k, cond in cond_grid_params.items():
        result = [x+[y] for x in result for y in _tuple(cond(pars(x)))]
    for prod in result:
        yield prod

def cond_dict_values(value, p):
    dat = []
    for key in value["keys"]:
        k1, k2 = key.split(" ")
        dat.append(p[k1][k2])
    for val, data in zip(value["vals"], value["data"]):
        if list(val) == list(dat):
            return data        

def parse_input_dict(params):
    if isinstance(params["input"], str):
        with open(params["input"], "rb") as f:
            params["input"] = pickle.load(f)
    grid_params = {}
    cond_grid_params = {}
    for module, mod_params in params.items():
        for key, value in mod_params.items():
            if isinstance(value, dict):
                cond_grid_params[module + " " + key] = (lambda p, v=value:
                    cond_dict_values(v, p))
                continue
            try:
                if isinstance(value, str):
                    continue
                iter(value)
                value = tuple(value) 
                params[module][key] = value
                grid_params[module + " " + key] = value 
            except TypeError:
                if callable(value):
                    cond_grid_params[module + " " + key] = value
    keys = list(grid_params.keys()) + list(cond_grid_params.keys())
    d = {"keys": keys, "vals": [], "data": []} 
    for item in product(grid_params, cond_grid_params):
        d["vals"].append(item)
        p = copy.deepcopy(params)
        for i, key in enumerate(keys):
            k1, k2 = key.split(" ")
            p[k1][k2] = item[i]
        d["data"].append(p)
    return d

test_mats = {}
test_vecs = {}

def get_matrix(idx, test_set=None):
    test_set = test_set or "default"
    name = "mat_" + test_set
    global test_mats
    if name not in test_mats:
        with open(os.path.join(TEST_DATA, name + ".pkl"), "rb") as f:
            test_mats[name] = pickle.load(f)
    return test_mats[name][idx]
    
def get_vector(idx, test_set=None):
    test_set = test_set or "default"
    name = "vec_" + test_set
    global test_vecs
    if name not in test_vecs:
        with open(os.path.join(TEST_DATA, name + ".pkl"), "rb") as f:
            test_vecs[name] = pickle.load(f)
    return test_vecs[name][idx]

def array_to_list(ar):
    if ar.dtype == complex:
        ar = [ar.real.tolist(), ar.imag.tolist()]
        return ar
    else:
        return ar.tolist()

def clean_params(params):
    ret = {}
    for module, m_params in params.items():
        ret[module] = {}
        for key, val in m_params.items():
            if isinstance(val, np.int64):
                val = int(val)
            if isinstance(val, np.ndarray):
                val = array_to_list(val)
            ret[module][key] = val
    return ret
                
def hash_params(params):
    params = clean_params(params)
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest(), params

def load_or_run(params, force=False):
    check_dirs()
    ha, cp = hash_params(params)
    if (not os.path.exists(os.path.join(BASE_DIR, INFO_DIR, ha))) or force:
        matrix = params["input"]["matrix"]
        vector = params["input"]["vector"]
        del params["input"]
        res = run_algorithm(params, (matrix, vector))
        with open(os.path.join(BASE_DIR, INFO_DIR, ha), "w") as f:
            f.write(json.dumps(cp, sort_keys=True, indent=2))
        with open(os.path.join(BASE_DIR, RAW_DIR, ha), "wb") as f:
            pickle.dump(res, f)
    else:
        with open(os.path.join(BASE_DIR, RAW_DIR, ha), "rb") as f:
            res = pickle.load(f)
    return res

def run_test(params, force=False):
    test_set = None
    problem_size = None
    if "test_set" in params["input"]:
        test_set = params["input"]["test_set"]
        del params["input"]["test_set"]
    if "n" in params["input"]:
        problem_size = params["input"]["n"]
        del params["input"]["n"]
    if isinstance(params["input"]["matrix"], int):
        matrix = get_matrix(params["input"]["matrix"], test_set)
    else:
        matrix = params["input"]["matrix"]
    if isinstance(params["input"]["vector"], int):
        vector = get_vector(params["input"]["vector"], test_set)
    else:
        vector = params["input"]["vector"]
    params["input"]["matrix"] = matrix
    params["input"]["vector"] = vector
    return load_or_run(params, force)

def filter_interests(result, interest):
    if interest == -1:
        return result

def run_tests(input_params, force=False, status=None, interest=None):
    d = parse_input_dict(input_params)
    l = len(d["vals"])
    c = 0
    ret = copy.deepcopy(d)
    ret["data"] = []
    for vals, params in zip(d["vals"], d["data"]):
        c += 1
        print("Test Run %d/%d, Parameter:\n" % (c, l), 
                *map(lambda x: "{:<25} {:>8}\n".format(x[0]+":", x[1]), zip(d["keys"], vals)))
        result = run_test(params, force)
        if status:
            print(status(result))
        if not interest is None:
            ret["data"].append(filter_interests(result, interest))
    return ret

def run_tests_from_file(path, force=False, status=None, interest=-1):
    with open(path, "r") as f:
        d = f.read()
    params = eval(d)
    return run_tests(params, force, status, interest)

print(run_tests_from_file("config_lookup.py"))

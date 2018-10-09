import numpy as np
from qiskit_aqua import run_algorithm
from generate_test_objects import generate_input, save_generated_inputs

import os
import itertools
import copy
import pickle
import hashlib
import json

BASE_DIR = "data"

RAW_DIR = "raw"
INFO_DIR = "info"

TEST_DATA = "test_objects"

test_mats = {}
test_vecs = {}

def get_matrix(idx, test_set=None):
    """ get matrix from test_set at pos idx """
    test_set = test_set or "default"
    name = "mat_" + test_set
    global test_mats
    if name not in test_mats:
        with open(os.path.join(TEST_DATA, name + ".pkl"), "rb") as f:
            test_mats[name] = pickle.load(f)
    return test_mats[name][idx]
    
def get_vector(idx, test_set=None):
    """ get vector from test_set at pos idx """
    test_set = test_set or "default"
    name = "vec_" + test_set
    global test_vecs
    if name not in test_vecs:
        with open(os.path.join(TEST_DATA, name + ".pkl"), "rb") as f:
            test_vecs[name] = pickle.load(f)
    return test_vecs[name][idx]


def check_dirs():
    """ check if data directories exists and create if not so """
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    if not os.path.exists(os.path.join(BASE_DIR, INFO_DIR)):
        os.mkdir(os.path.join(BASE_DIR, INFO_DIR))
    if not os.path.exists(os.path.join(BASE_DIR, RAW_DIR)):
        os.mkdir(os.path.join(BASE_DIR, RAW_DIR))

def _gen_dict(keys, values, static_keys, static_values):
    """ generate dict for conditional params sets """
    ret = {}
    for key, val in zip(static_keys+keys, static_values+values):
        k1, k2 = key.split(" ")
        if k1 not in ret:
            ret[k1] = {}
        ret[k1][k2] = val
    return ret

def _tuple(obj):
    """ make a tuple out of obj """
    try:
        iter(obj)
        return tuple(obj)
    except TypeError:
        return (obj,)

def product(grid_params, cond_grid_params, static_keys, static_values):
    """ cartesian product of parameter grid + conditional parameters """
    pars = lambda x: _gen_dict(list(grid_params.keys())
            + list(cond_grid_params.keys()), x, static_keys, static_values)
    result = [[]]
    for pool in grid_params.values():
        result = [x+[y] for x in result for y in _tuple(pool)]
    for k, cond in cond_grid_params.items():
        result = [x+[y] for x in result for y in _tuple(cond(pars(x)))]
    for prod in result:
        yield prod

def cond_dict_values(value, p):
    """ function for conditional parameters specified through dict """
    dat = []
    for key in value["keys"]:
        k1, k2 = key.split(" ")
        dat.append(p[k1][k2])
    for val, data in zip(value["vals"], value["data"]):
        if list(val) == list(dat):
            return data        

def parse_input_dict(params):
    """ parse the parameters dict with specified grid """
    if "file" in params["input"]:
        with open(params["input"]["file"], "rb") as f:
            x = pickle.load(f)
            params["input"]["matrix"] = x["matrix"]
            params["input"]["vector"] = x["vector"]
        del params["input"]["file"]
    grid_params = {}
    cond_grid_params = {}
    static_keys = []
    static_values = []
    for module, mod_params in params.items():
        for key, value in mod_params.items():
            if isinstance(value, dict):
                cond_grid_params[module + " " + key] = (lambda p, v=value:
                    cond_dict_values(v, p))
                continue
            try:
                if isinstance(value, str):
                    raise TypeError()
                iter(value)
                value = tuple(value) 
                params[module][key] = value
                grid_params[module + " " + key] = value 
            except TypeError:
                if callable(value):
                    cond_grid_params[module + " " + key] = value
                else:
                    static_keys.append(module + " " + key)
                    static_values.append(value)
    keys = list(grid_params.keys()) + list(cond_grid_params.keys())
    d = {"keys": keys, "vals": [], "data": []} 
    for item in product(grid_params, cond_grid_params, static_keys,
            static_values):
        d["vals"].append(item)
        p = copy.deepcopy(params)
        for i, key in enumerate(keys):
            k1, k2 = key.split(" ")
            p[k1][k2] = item[i]
        d["data"].append(p)
    return d


def array_to_list(ar):
    """ make np.ndarrays json parseable """
    if ar.dtype == complex:
        ar = [ar.real.tolist(), ar.imag.tolist()]
        return ar
    else:
        return ar.tolist()

def clean_params(params):
    """ make params json parseable """
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
    """ has paramters """
    params = clean_params(params)
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest(), params

def load_or_run(params, force=False):
    """ try to find saved run for parameters, otherwise run and save """
    check_dirs()
    ha, cp = hash_params(params)
    if (not os.path.exists(os.path.join(BASE_DIR, INFO_DIR, ha))) or force:
        matrix = params["input"]["matrix"]
        vector = params["input"]["vector"]
        del params["input"]
        save_generated_inputs()
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
    """ run HHL with parameters params, needs additional input section """
    test_set = None
    problem_size = None
    if "type" in params["input"] and params["input"]["type"] == "generate":
        params["input"] = generate_input(params)
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
    """ filter for interests in return data """
    if interest == -1:
        return result
    elif isinstance(interest, str):
        return result[interest]
    elif callable(interest):
        return interest(result)
    else:
        ret = []
        for inter in interest:
            if isinstance(inter, str):
                ret.append(result[inter])
            elif callable(inter):
                ret.append(inter(result))
            else:
                d = result
                for i in inter:
                    d = d[i]
                ret.append(d)
        return ret

def fmt(x, input_params):
    t, v = x
    if isinstance(v, np.ndarray):
        k1, k2 = t.split(" ")
        d = input_params[k1][k2]
        i = 0
        for dd in d:
            if np.allclose(dd, v, 1e-10):
                v = i
                break
            i += 1
        v = i
    s = "{:<25} {:>8}\n".format(t+":", str(v))
    return s

def run_tests(input_params, force=False, status=None, interest=None):
    """ run params with specified parameter grid """
    d = parse_input_dict(input_params)
    l = len(d["vals"])
    c = 0
    ret = copy.deepcopy(d)
    ret["data"] = []
    for vals, params in zip(d["vals"], d["data"]):
        c += 1
        print("Test Run %d/%d, Parameter:\n" % (c, l), 
                *map(lambda x: fmt(x, input_params), zip(d["keys"], vals)))
        result = run_test(params, force)
        if status:
            print(status(result))
        if not interest is None:
            ret["data"].append(filter_interests(result, interest))
    return ret

def run_tests_from_file(path, force=False, status=None, interest=-1):
    """ run_test with params = eval(path) """
    with open(path, "r") as f:
        d = f.read()
    params = eval(d)
    return run_tests(params, force, status, interest)

def get_for(params, data):
    """
    Extracts information for specifically set parameters from data dictionary
    Args:
        params: dictionary of wished parameters, e.g. {'eig num_ancillae': 4}
        data: the data dictionary
        subs: the argument for a result dictionary, e.g. 'fidelity'
    Returns:
        Stripped down data dictionary or list of result dictionarys if only one
        free parameter left.
    """
    ret = {"vals": [], "data": []}
    idxs = list(map(lambda x: data["keys"].index(x), params.keys()))
    vals = list(map(lambda x: params[x], params.keys()))
    ret["keys"] = [key for key in data["keys"] if key not in params]
    for v, res in zip(data["vals"], data["data"]):
        cont = True
        for x, y in zip(vals, [v[i] for i in idxs]):
            if isinstance(x, (tuple, list)):
                x = tuple(x)
            else:
                x = (x,)
            if not y in x:
                cont = False
                break
        if cont:
            ret["vals"].append([v[i] for i in range(len(v)) if i not in idxs])
            ret["data"].append(res)
    if len(ret["keys"]) == 1:
        return ret["data"]
    return ret

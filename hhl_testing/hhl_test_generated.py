from hhl_test_suite import run_test, run_tests, get_matrix, get_vector, get_for
import numpy as np
import matplotlib.pyplot as plt
import pickle

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": 50,
        "expansion_mode": "suzuki",
        "expansion_order": 2,
        "negative_evals": False,
        "num_ancillae": (6, 7, 8, 9),
    },
    "reciprocal": {
        "name": "GENCIRCUITS",
        "lambda_min": 0.9,
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    },
    "input": {}
}

with open("test_objects/specified.pkl", "rb") as f:
    input_data = pickle.load(f)


data = {}
for ne in (True, False):
    data[ne] = {}
    params["eigs"]["negative_evals"] = ne
    for n in (2, 4):
        data[ne][n] = []
        params["input"]["n"] = n
        c = 0
        for mat, vecs in input_data[ne][n]:
            c += 1
            params["input"]["matrix"] = mat
            cc = 0
            for vec in vecs:
                cc += 1
                params["input"]["vector"] = vec
                print("\n(negative_evals,", ne, ") (n,", n, ") (matrix,", "%d/%d"
                        % (c, len(input_data[ne][n])), ") (vector, ", "%d/%d"
                        % (cc, len(vecs)), ")")
                data[ne][n].append(run_tests(params))

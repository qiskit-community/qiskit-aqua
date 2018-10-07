from hhl_test_suite import run_test, run_tests, get_matrix, get_vector, get_for
import numpy as np
import matplotlib.pyplot as plt 

params = {
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": (4, 10),
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
        "n": 2,
        "matrix": 0,
        "vector": 1,
    }
}

res = run_test(params)
sol = np.linalg.solve(res["matrix"], res["invec"]) 
sol = sol/np.linalg.norm(sol)
invec = res["invec"]
r = res["matrix"].dot(res["result"])
print(res["fidelity"])
print(invec[0]/r[0]*r)
print(invec)

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
        "num_time_slices": 1,
        "expansion_mode": "suzuki",
        "expansion_order": 2,
        "negative_evals": True,
        "num_ancillae": 5,
    },
    "reciprocal": {
        "name": "GENCIRCUITS",
        "lambda_min": 1,
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
print(invec[0]/r[0]*r)
print(invec)

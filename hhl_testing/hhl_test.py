from hhl_test_suite import run_tests, get_matrix, get_vector, get_for
import numpy as np
import matplotlib.pyplot as plt 

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
        "n": 2,
        "matrix": 0,
        "vector": 1,
    }
}

res = run_tests(params, status=lambda res: print(res["fidelity"]))

for i in params["eigs"]["num_ancillae"]:
    data = np.array(get_for({"eigs num_ancillae": i}, res, "fidelity"))
    plt.plot(params["eigs"]["num_time_slices"], 1-data)
plt.yscale('log')
plt.show()

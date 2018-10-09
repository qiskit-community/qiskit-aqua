from hhl_test_suite import run_tests, get_for
import numpy as np
import matplotlib.pyplot as plt

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
        "num_ancillae": 6
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
        "n": (2, 4),
        "type": "generate",
        "test_set": "general_condition",
        "repetition": 0,
        "condition": range(10, 110, 20)
    }
}
res = run_tests(params, interest="fidelity")
#
# x = np.arange(10, 110, 10)
# for num_ancillae in (6, 7, 8):
#     fids = []
#     for rep in range(2):
#         fids.append(np.array(get_for({"reciprocal pat_length": 5, "eigs num_ancillae":
#             num_ancillae, "input repetition": rep}, res)))
#     y = np.mean(fids, axis=0)
#     e = np.std(fids, axis=0)
#     plt.plot(x, fids[0])
#     plt.plot(x, fids[1])
#     plt.errorbar(x, y, yerr=e, fmt="o")
#     plt.show()

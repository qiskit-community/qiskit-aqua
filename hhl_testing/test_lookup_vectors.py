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
        "pat_length": 5
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    },
    "input": {
        "n": 2,
        "type": "generate",
        "test_set": "general_condition",
        "repetition": range(10),
        "condition": 50,
        "vector": [np.array((np.sin(x), np.cos(x))) for x in np.linspace(0,
            2*np.pi, 20, endpoint=False)]
    }
}
res = run_tests(params, interest=["fidelity", "matrix", "probability"])

data = []
data_prob = []
matrix = None
for i in params["input"]["repetition"]:
    dat = []
    dat_prob = []
    for x in get_for({"input repetition": i}, res):
        dat.append(x[0])
        dat_prob.append(x[2])
        if matrix is None: matrix = x[1]
    data.append(np.array(dat))
    data_prob.append(np.array(dat_prob))
data = np.array(data)
data_prob = np.array(data_prob)
x = np.linspace(0, 2*np.pi, 20, endpoint=False)

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

mean_prob = np.mean(data_prob, axis=0)
std_prob = np.std(data_prob, axis=0)

# plt.plot(x, mean)
fig, ax1 = plt.subplots()

ax1.errorbar(x, mean, yerr=std, fmt="o", capsize=5)
ax2 = ax1.twinx()
ax2.errorbar(x, mean_prob, yerr=std_prob, fmt="or", capsize=5)

ax1.grid()
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], [0, "$\pi/2$", "$\pi$", "$3\pi/2$"])

plt.show()



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

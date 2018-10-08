from hhl_test_suite_2 import run_tests, get_for
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
        "num_ancillae": 8
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
        "type": "generate",
        "test_set": "specified",
        "repetition": 1,
        "n": 2,
        "cond": (2,)
    }
}



# cond = (2, 3, 5, 8, 10, 30, 50, 100)
res = run_tests(params, interest=["eigenvalues", "fidelity"])

print(res)
# for n in params["eigs"]["num_ancillae"]:
#     invecs = [3*i for i in range(len(cond))]
#
#     fids = get_for({"eigs num_ancillae": n, "input vector": invecs}, res)
#     plt.plot(cond, fids, label="num_ancillae=%d"%n, marker="o")
#     # plt.ylim((0.9, 1.001))
# plt.legend()
# plt.xlabel("condition number")
# plt.ylabel(r"fidelity $|\langle \tilde{x}|x\rangle|^2$")
# plt.title("Fidelities depending on condition number")
# plt.grid(True)
# plt.savefig("plots/fid_cond.pdf")


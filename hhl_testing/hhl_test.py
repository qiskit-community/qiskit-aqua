from hhl_test_suite import run_tests_from_file, get_for
import numpy as np
import matplotlib.pyplot as plt

res = run_tests_from_file("config_lookup.py", interest="fidelity")

x = np.arange(10, 110, 10)
for num_ancillae in (6, 7, 8):
    fids = []
    for rep in range(2):
        fids.append(np.array(get_for({"reciprocal pat_length": 5, "eigs num_ancillae":
            num_ancillae, "input repetition": rep}, res)))
    y = np.mean(fids, axis=0)
    e = np.std(fids, axis=0)
    plt.plot(x, fids[0])
    plt.plot(x, fids[1])
    plt.errorbar(x, y, yerr=e, fmt="o")
    plt.show()



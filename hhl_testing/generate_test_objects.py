from qiskit_aqua.utils import random_hermitian
import numpy as np
import pickle

cond = (2, 3, 5, 8, 10, 30, 50, 100)

data = {}

for ne in [True, False]:
    data[ne] = {}
    for n in [2, 4]:
        data[ne][n] = []
        for c in cond:
            v = []
            h = random_hermitian(n, K=(c, 1, -1 if ne else 1))
            data[ne][n].append((h, []))
            w = np.linalg.eigh(h)[1]
            v.append(w.dot(np.ones(n)))
            if n == 2:
                v.append(w.dot(np.array([0.8, 0.2])))
                v.append(w.dot(np.array([0.2, 0.8])))
            for vi in v:
                data[ne][n][-1][1].append(vi)
print(data)
with open("test_objects/specified.pkl", "wb") as f:
    pickle.dump(data, f)

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.visualization import plot_circuit
import numpy as np
import ncx

import matplotlib.pyplot as plt

def _construct_circuit(qc, inreg):
    out = QuantumRegister(int(np.ceil(np.log2(len(inreg))))+1)
    anc = QuantumRegister(1)
    qc.add(out, anc)
    for i in range(len(inreg)):
        binary = np.binary_repr(len(inreg)-i).rjust(len(out), "0")
        qc.x(anc)
        for j, b in enumerate(binary):
            if b == "1":
                qc.ccx(inreg[i], anc[0], out[j])
        qc.x(anc)
        for j, b in enumerate(binary):
            if b == "0":
                qc.x(out[j])
        qc.ncx(out, anc[0])
        for j, b in enumerate(binary):
            if b == "0":
                qc.x(out[j])
    qc.x(inreg)
    for i in range(len(inreg)):
        qc.x(inreg[i])
        qc.ncx(inreg, anc[0])
        qc.x(inreg[i])
    qc.x(inreg)
    qc.x(anc)
    for i in reversed(range(len(out))):
        qc.ncx([anc[0]] + [out[j] for j in range(len(out)) if i < j], out[i])
    qc.x(anc)
    return out, anc

def prepare(val, size=None):
    if not size:
        size = int(np.ceil(np.log2(val))) + (1 if np.log2(val)%1 == 0 else 0)
    q = QuantumRegister(size)
    qc = QuantumCircuit(q)
    binary = np.binary_repr(val).rjust(len(q), "0")
    for j, b in enumerate(binary):
        if b == "1":
            qc.x(q[j])
    return qc, q

size = 5
x = np.arange(1, 2**size)
y = []
for i in x:
    qc, q = prepare(i, 5)

    out, anc = _construct_circuit(qc, q)

    # if i == 1:
    #     plot_circuit(qc)

    c = ClassicalRegister(len(out))
    c2 = ClassicalRegister(1)
    qc.add(c, c2)
    qc.measure(out, c)
    qc.measure(anc, c2)
    
    print(i)

    res = execute(qc, "local_qasm_simulator", shots=10).result()
    bitstr = "".join(reversed(list(res.get_counts().keys())[0][2:]))
    y.append(int(bitstr, 2))


plt.plot(x, y)
plt.plot(x, np.ceil(np.log2(x)))
plt.show()

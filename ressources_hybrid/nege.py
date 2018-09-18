from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.tools.visualization import plot_circuit
import numpy as np

def qft(self, q):
    """QFT on q."""
    for j in range(len(q)):
        for k in range(j):
            self.cu1(np.pi / float(2**(j - k)), q[j], q[k])
        self.h(q[j])

def iqft(self, q):
    """inverse QFT on q."""
    for j in reversed(range(len(q))):
        self.h(q[j])
        for k in reversed(range(j)):
            self.cu1(-np.pi / float(2**(j - k)), q[j], q[k])

QuantumCircuit.qft = qft
QuantumCircuit.iqft = iqft

def clean_register(q):
    qc = QuantumCircuit(q)
    sgn = q[0]
    qs = [q[i] for i in range(1, len(q))]
    for i in range(len(qs)):
        qc.cx(sgn, qs[i])
    qc.qft(qs)
    for i in range(len(qs)):
        qc.cu1(2*np.pi/2**(i+1), sgn, qs[len(qs)-i-1])
    qc.iqft(qs)
    return qc


def maxminus(x):
    ret = ""
    flag = False
    for e in reversed(x):
        if e == "0":
            if flag:
                ret = "1"+ret
                flag = True
            else:
                ret = "0"+ret
                flag = False
        else:
            if flag:
                ret = "0"+ret
                flag = True
            else:
                ret = "1"+ret
                flag = True
    return ret

def m2(x):
    ret = ""
    for e in reversed(x):
        ret = ("1" if e == "0" else "0") + ret
    return ret

def encode_num(x, width=None):
    s = np.binary_repr(x)
    if width:
        s = s.rjust(width, "0")
    return s

def decode_num(bitstr):
    return int(bitstr, 2)

n = 5
for j in range(2**n):
    q = QuantumRegister(n)
    c = ClassicalRegister(n)
    qc = QuantumCircuit(q, c)

    s = encode_num(j, width=n)
    for i, x in enumerate(s):
        if x == "1":
            qc.x(q[i])

    qc += clean_register(q)

    qc.measure(q, c)

    res = execute(qc, "local_qasm_simulator").result()
    k = list(res.get_counts().keys())[0][:-1]
    s = "".join(reversed(k))

    print(j, s, decode_num(s))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.extensions.simulator import snapshot
import numpy as np

c = 2
m = 2
l = 2

C = QuantumRegister(c)
M = QuantumRegister(m)
L = QuantumRegister(l)

qc = QuantumCircuit(C, M, L)

t0 = 2*np.pi

'''Part1
qc.h(M)
qc.h(L)
for i in range(m):
    qc.u1(t0/2**(i+1), M[m-i-1])

res = execute(qc, "local_statevector_simulator").result()
sv = res.get_data()["statevector"]

svtheo_p = np.array([np.exp(1j*t0*p/2**m)/np.sqrt(2**m) for p in range(0, 2**m)])
svtheo_s = np.ones(2**l)/np.sqrt(2**l)
svtheo = np.kron(svtheo_p, svtheo_s)

def uv(k, n):
    x = np.zeros(2**n)
    x[k] = 1
    return x

svt2 = 1/np.sqrt(2**(m+l))*sum([sum([np.exp(1j*t0*p/2**m)*np.kron(uv(s, l),
    uv(p, m)) for p in
    range(2**m)]) for s in range(2**l)])

print(sv.dot(svt2.conj()))
'''

def ccz(lam, ctl1, ctl2, tgt, qc):
    qc.ccx(ctl1, ctl2, tgt)
    qc.u1(-lam/2.0, tgt)
    qc.ccx(ctl1, ctl2, tgt)
    qc.u1(lam/2.0, tgt)
    qc.cu1(lam/2, ctl1, ctl2)

def H0m(m1, t0, ctl1, ctl2, treg, qc):
    for i in range(m1):
        ccz(t0/2**(i+1), ctl1, ctl2, treg[i], qc)

def G(k, t, t0, L, M, C, qc):
    for i in range(len(C)):
        H0m(t+k)


qc.x(M[0])
qc.x(C[0])
qc.x(L[0])

ccz(1, C[0], L[0], M[0], qc)

qc.snapshot(1)
print(qc.data)

res = execute(qc, "local_qasm_simulator", config={"data":
    ["quantum_state_ket"]}, shots=1).result()
print(res.get_data()["snapshots"]["1"]["quantum_state_ket"])


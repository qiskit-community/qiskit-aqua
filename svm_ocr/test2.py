import numpy as np
from qiskit_aqua.hhl import QPE
from qiskit import execute, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import plot_circuit
from copy import deepcopy

def cry(theta, ctl, tgt, qc):
    if theta == 0 or np.isnan(theta): return
    qc.h(tgt)
    qc.s(tgt)
    qc.cx(ctl, tgt)
    qc.u1(theta/2, tgt)
    qc.cx(ctl, tgt)
    qc.u1(-theta/2, tgt)
    qc.sdg(tgt)
    qc.h(tgt)

def ccry(theta, ctl1, ctl2, tgt, qc):
    if theta == 0 or np.isnan(theta): return
    qc.h(tgt)
    qc.s(tgt)
    qc.ccx(ctl1, ctl2, tgt)
    qc.u1(theta/2, tgt)
    qc.ccx(ctl1, ctl2, tgt)
    qc.u1(-theta/2, tgt)
    qc.sdg(tgt)
    qc.h(tgt)


matrix = np.array([[0.49997724, 0.2491572 ], [0.2491572,  0.50002276]])
#matrix = np.array([[1, 0], [0,  2]])

print(np.linalg.eig(matrix))

n = 2
params = {
        "algorithm": {
            "num_ancillae": n,
            "num_time_slices": 5,
            "evo_time": 2*np.pi
            },
        "initial_state": {
            "name": "CUSTOM",
            "state_vector": [1/2**0.5, -1/2**0.5]
            }
        }


qpe = QPE()
qpe.init_params(params, matrix) 
data = qpe._compute_eigenvalue()
print(data)


def create_circuit(b="Z"):
    qpe = QPE()
    qpe.init_params(params, matrix) 
    qc = qpe._setup_qpe()
    print(2*np.pi/qpe._evo_time)
    a = qc.regs["a"]
    q = qc.regs["q"]
    x = QuantumRegister(1)
    qc.add(x)

    #cry(2*np.arcsin(1), a[1], x[0], qc)
    #cry(2*(-np.arcsin(1)+np.arcsin(1/3)), a[0], x[0], qc)

    #cry(2*np.pi/4, a[1], x[0], qc)
    #cry(2*np.pi/8, a[0], x[0], qc)

    qc.x(a[0])
    t = 2*np.pi/qpe._evo_time
    print(t)
    ccry(2*np.arcsin(1/(1*t)), a[0], a[1], x[0], qc)
    qc.x(a)
    ccry(2*np.arcsin(1/(2*t)), a[0], a[1], x[0], qc)
    qc.x(a[1])
    ccry(2*np.arcsin(1/(3*t)), a[0], a[1], x[0], qc)
    
    print("-")
    print(2*np.arcsin(1/(3*t)))
    print(2*np.arcsin(1/(2*t)))
    print(2*np.arcsin(1/(1*t)))

    qc += qpe._construct_inverse()

    x1 = np.array([0.997, 0.159])
    x2 = np.array([0.354, 0.935])

    t1 = 2*np.arctan(0.159/0.987)
    t2 = 2*np.arctan(0.935/0.354)
    
    bas = [0.072/0.997, 0.941/0.338]
    
    t0 = 2*np.arctan(bas[1])
    print(t0)
    
    qc.x(q[0])
    ccry(t1, x[0], q[0], a[0], qc)
    qc.x(q[0])
    ccry(t2, x[0], q[0], a[0], qc)
    
    cry(-t0, x[0], a[0], qc)
    qc.ch(x[0], q[0])
    #

    xcr = ClassicalRegister(1)
    qcr = ClassicalRegister(1)
    acr = ClassicalRegister(1)
    qc.add(xcr, qcr, acr)
    qc.measure(x[0], xcr[0])
    qc.measure(q[0], qcr[0])
    qc.measure(a[0], acr[0])
    res = execute(qc, "local_qasm_simulator", shots=1024).result()
    print(res)
    print(res._result)
    counts = res.get_counts()
    print(counts)


    #qcr = ClassicalRegister(2)
    #qc.add(qcr)
    #if b == "X":
    #    qc.h(q[0])
    #if b == "Y":
    #    qc.sdg(q[0])
    #    qc.h(q[0])
    #qc.measure(q[0], qcr[0])
    #qc.measure(x[0], qcr[1])

    #res = execute(qc, "local_qasm_simulator", shots=3*1024).result()
    #counts = res.get_counts()
    #print(counts)
    #d = [0, 0]
    #for k, v in counts.items():
    #    if k[0] == "1":
    #        d[int(k[1])] = np.sqrt(v)
    #d = np.array(d)
    #d = d/np.sqrt(d.dot(d))
    #return d

create_circuit()

#d1 = create_circuit()
#d2 = create_circuit(b="X")
#
#dist = []
#vv = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
#for v in vv:
#    dist += [np.linalg.norm(np.abs(np.array([[1, 1], [1,
#        -1]]).dot(np.diag(d1)).dot(np.array(v)))-d2)]
#
#d = d1*np.array(vv[np.argmin(dist)])
#d = d/np.sqrt(d.dot(d))
#
#print(matrix)
#print(matrix.dot(d))
#
#req = np.linalg.inv(matrix).dot(np.array([1, 0]))
#req = req/np.sqrt(req.dot(req))
#print(d)
#print(req)
#print(req.dot(d))

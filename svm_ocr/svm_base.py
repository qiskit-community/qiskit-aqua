from qiskit import QuantumProgram, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.tools.visualization import plot_circuit

from qiskit_aqua.hhl import QPE

import qiskit.tools.qcvv.tomography as tomo

import numpy as np

def _construct_density_matrix2x2(training_data, backend="local_qasm_simulator",
        shots=1024):
    ts = [2*np.arctan(v[0][1]/v[0][0]) for v in training_data]
    print("Construct Matrix via Tomography")

    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.x(q[0])
    qc.cu3(ts[0], 0, 0, q[0], q[1])
    qc.x(q[0])
    qc.cu3(ts[1], 0, 0, q[0], q[1])

    tomo_set = tomo.state_tomography_set([0])

    qp = QuantumProgram()
    qp.add_circuit('density_matrix', qc)

    tomo_circuit_names = tomo.create_tomography_circuits(qp, 'density_matrix',
            q, c, tomo_set)
    result = qp.execute(tomo_circuit_names, backend=backend, shots=shots)
    data = tomo.tomography_data(result, 'density_matrix', tomo_set)
    
    # mat = np.array([[np.array(v).dot(np.array(w)) for v in invecs] for w in
    #     invecs])
    # print(mat/np.trace(mat))

    matrix = tomo.fit_tomography_data(data)
    print(matrix)
    return matrix

def ncx(ctls, tgt, qc):
    if len(ctls) == 1:
        return qc.cx(ctls[0], tgt)
    if len(ctls) == 2:
        return qc.ccx(*ctls, tgt)
    else:
        print("ncx %d not yet implemented!" % len(ctls))

def ncry(theta, ctls, tgt, qc, beg=True, end=True):
    if beg: qc.rx(np.pi/2, tgt)
    ncx(ctls, tgt, qc)
    qc.u1(theta/2, tgt)
    ncx(ctls, tgt, qc)
    qc.u1(-theta/2, tgt)
    if end: qc.rx(-np.pi/2, tgt)
    

def rotation(qc, evo_time, n):
    a = QuantumRegister(1, name="anc")
    qc.add(a)
    e = qc.regs["eigs"]
    a = a[0]
    e = [e[i] for i in range(n)]
    for k in range(2**n):
        if k > 0:
            s = "{:b}".format(k).rjust(n, "0")
            for i in range(n):
                if s[i] == "0":
                    qc.x(e[i])
            t = 2*np.pi/evo_time
            ncry(2*np.arcsin(1/(k*t)), e, a, qc, beg=k==1, end=k==2**n-1)
            for i in range(n):
                if s[i] == "0":
                    qc.x(e[i])
    return qc

def _construct_hhl(matrix, num_time_slices=1, num_ancilla=2, evo_time=None,
        invec=[1/2**0.5, -1/2**0.5], same_settings_qpe_only=False,
        expansion_order=1, backend="local_qasm_simulator"):
    params = {
        "algorithm": {
            "num_ancillae": num_ancilla,
            "num_time_slices": num_time_slices,
            "expansion_mode": "trotter" if expansion_order == 1 else "suzuki",
            "expansion_order": expansion_order,
            "evo_time": evo_time,
            "backend": backend,
            },
        "initial_state": {
            "name": "CUSTOM",
            "state_vector": invec
            }
        }
    if same_settings_qpe_only:
        qpe2 = QPE()
        qpe2.init_params(params, matrix)
        res = qpe2.run()
        print(res)
    qpe = QPE()
    qpe.init_params(params, matrix)

    qc = qpe._setup_qpe()

    evo_time = evo_time or qpe._evo_time
    print(evo_time)

    rotation(qc, evo_time, num_ancilla)

    qc += qpe._construct_inverse()
    return qc

def _construct_svm_expectation_value2x2(qc, training_data, testvec):
    ts = [2*np.arctan(v[0][1]/v[0][0]) for v in training_data]
    t = 2*np.arctan(testvec[1]/testvec[0])

    a = qc.regs["eigs"]
    b = qc.regs["comp"]
    x = qc.regs["anc"]

    c = ClassicalRegister(1, name="expv")
    cx = ClassicalRegister(1, name="canc")
    qc.add(c, cx)

    qc.h(a[1])

    qc.x(b[0])
    ncry(ts[0], [a[1], b[0]], a[0], qc, end=False)
    qc.x(b[0])
    ncry(ts[1], [a[1], b[0]], a[0], qc, beg=False)
    ncry(t, [a[1]], a[0], qc)
    
    qc.h(a[1])

    qc.measure(a[1], c[0])
    qc.measure(x, cx)

    return qc

def construct_full_svms(training_data, test_data, matrix=None, evo_time=None,
        time_slices=1, expansion_order=1, backend="local_qasm_simulator"):
    '''
    creates svm circuits for a set of test_data vectors.
    '''
    if matrix is None:
        matrix = _construct_density_matrix2x2(training_data, backend=backend)
    qcs = []
    for testvec in test_data:
        invec = normalize([x[1] for x in training_data])
        qc = _construct_hhl(matrix, evo_time=evo_time, invec=invec,
                num_time_slices=time_slices, expansion_order=expansion_order,
                backend=backend)
        _construct_svm_expectation_value2x2(qc, training_data, testvec)
        qcs.append(qc)
    return qcs

def classify_results(result, qcs, test_data, debug=False):
    classes = []
    for qc, vec in zip(qcs, test_data):
        counts = result.get_counts(qc)
        if debug: print(vec, counts)
        t = 0
        for k, v in counts.items():
            if k[0] == "1":
                t += v*(-1 if k[-1] == "1" else 1)
        if t == 0:
            t = 1
        classes.append(int(t/abs(t)))
    return list(zip(test_data, classes))

def classify(training_data, test_data, matrix=None, evo_time=None,
        time_slices=1, expansion_order=1, backend="local_qasm_simulator",
        shots=1024, debug=False):
    qcs = construct_full_svms(training_data, test_data, matrix=matrix,
            evo_time=evo_time)
    res = execute(qcs, backend, shots=shots).result()
    return classify_results(res, qcs, test_data, debug=debug)

def normalize(vec):
    vec = np.array(vec)
    return vec/np.sqrt(vec.dot(vec.conj()))

if __name__ == "__main__":
    from qiskit import register
    import Qconfig
    #register(Qconfig.APItoken)
    matrix = np.array([[0.49997724, 0.2491572 ], [0.2491572,  0.50002276]])
    x1 = [0.997, 0.159]
    x2 = [0.354, 0.935]
    training_data = [(x1, 1), (x2, -1)]
    test_data = [(0.997, -0.072), (0.338, 0.941)]
    print(classify(training_data, test_data, matrix=matrix, evo_time=2*np.pi))

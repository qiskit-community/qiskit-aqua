#!/usr/bin/env python -W ignore::DeprecationWarning
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import qiskit.extensions.simulator
from qiskit.extensions.simulator.snapshot import Snapshot
from qiskit.tools.visualization import plot_circuit
import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy.combinatorics.graycode import GrayCode

current_slot = 0
slots = []

def get_slot():
    global current_slot, slots
    x = current_slot
    slots += [str(x)]
    current_slot += 1
    return x

def ncu1(qc, lam, regs, tgt):
    g = GrayCode(len(regs))[1:]
    
def ncx(qc, regs, tgt, phase=-1j):
    if len(regs) == 1:
        qc.cx(*regs, tgt)
    elif len(regs) == 2:
        qc.ccx(*regs, tgt)
    if phase:
        if len(regs) == 1:
            qc.u1(-abs(np.log(phase)), *regs)
        if len(regs) == 2:
            qc.cu1(-abs(np.log(phase)), *regs)

def input_gearbox(qc, vals, regs, tgt):
    cr = ClassicalRegister(len(regs))
    qc.add(cr)
    for q, v in zip(regs, vals):
        qc.rx(2*v, q)
    ncx(qc, regs, tgt)
    for q, v in zip(regs, vals):
        qc.rx(-2*v, q)
    qc.measure(regs, cr)
    qc.snapshot(get_slot())
    qc.reset(regs)

def input_par(qc, vals, regs, tgt):
    cr = ClassicalRegister(len(regs))
    qc.add(cr)
    for q, v in zip(regs, vals):
        qc.rx(2*v, q)
    ncx(qc, regs, tgt, phase=1j**(len(vals)-1))
    for i in range(len(regs)-1):
        qc.cx(regs[i+1], regs[i])
    qc.h(regs[len(regs)-1])
    qc.measure(regs, cr)
    qc.snapshot(get_slot())
    qc.reset(regs)

def cleanup_ket(ket):
    l = len(list(ket.keys())[0])-1
    x = "0"*l
    ret = {"1": 0, "0": 0}
    if "1"+x in ket:
        ret["1"] = ket["1"+x][0]+ket["1"+x][1]*1j
    if "0"+x in ket:
        ret["0"] = ket["0"+x][0]+ket["0"+x][1]*1j
    if "1"+x not in ket and "0"+x not in ket:
        return None
    return ret
       
def get_x_angle(cket):
    print(cket)
    if cket != None:
        a = cket["0"]
        b = cket["1"]
        if a == 0:
            a = 1e-16
        return np.arctan(-b.imag/a.real)

def display_in_terms_of_pi(angle):
    x = angle/np.pi
    print("%.3f Pi" % x)

def get_amp(angle, ret):
    a = np.array([np.cos(angle), -1j*np.sin(angle)])
    b = np.array([ret["0"], ret["1"]])
    return sum(b/a).real/2

def test_gearbox(vals, plot=False):
    q = QuantumRegister(len(vals)+1)
    c = ClassicalRegister(len(vals))

    qc = QuantumCircuit(q, c)

    input_gearbox(qc, vals, [q[i] for i in range(len(q)-1)],
            q[len(q)-1])
    
    if plot: plot_circuit(qc)
    qc.snapshot("1")
    for i in range(len(vals)):
        qc.measure(q[i], c[i])

    result = execute(qc, "local_qasm_simulator", config={"data":
        ["quantum_state_ket"]}, shots=1).result()
    res = cleanup_ket(result.get_snapshot("1").get("quantum_state_ket")[0])
    return get_x_angle(res)

def test_par(vals, plot=False):
    q = QuantumRegister(len(vals)+1)
    c = ClassicalRegister(len(vals))

    qc = QuantumCircuit(q, c)

    input_par(qc, vals, [q[i] for i in range(len(q)-1)],
            q[len(q)-1])
    
    if plot: plot_circuit(qc)
    qc.snapshot("1")
    for i in range(len(vals)):
        qc.measure(q[i], c[i])

    result = execute(qc, "local_qasm_simulator", config={"data":
        ["quantum_state_ket"]}, shots=1).result()
    res = cleanup_ket(result.get_snapshot("1").get("quantum_state_ket")[0])
    angle = get_x_angle(res)
    amp = get_amp(angle, res)
    return angle, amp

def full_test_gearbox():
    x = np.linspace(0, np.pi, 100)
    ret = []
    for xi in x:
        ret.append(test_gearbox([xi, xi]))
    plt.plot(x, ret)
    y = np.arctan(np.tan(np.arcsin(np.sin(x)*np.sin(x)))**2)
    plt.plot(x, y)
    plt.plot(x, x**4)
    plt.show()

def full_test_par():
    x = np.linspace(0, np.pi, 100)
    ret = []
    amps = []
    for xi in x:
        angle, amp = test_par([xi, xi])
        ret.append(-angle)
        amps.append(amp)
    plt.plot(x, ret)
    y = np.arctan(np.tan(x)*np.tan(x))
    plt.plot(x, y)
    plt.plot(x, x**2)
    plt.plot(x, amps)
    plt.show()
    
if __name__ == "__main__":
    ar = QuantumRegister(2)
    q = QuantumRegister(1)
    qc = QuantumCircuit(ar, q)

    input_par(qc, [0.1, 0.2], ar, q[0])
    input_gearbox(qc, [0.1, 0.2], ar, q[0])

    res = execute(qc, "local_qasm_simulator", config={"data":
        ["hide_statevector", "quantum_state_ket"]}, shots=20).result()

    qc.data = list(filter(lambda x: not isinstance(x, Snapshot), qc.data))
    #plot_circuit(qc)
    print(res.get_snapshot("1").get("quantum_state_ket"))


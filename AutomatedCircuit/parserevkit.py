from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt
import numpy as np
import math

def nc_toffoli(qc, qr1, qr2, ctl,tgt,ctlnumber,offset, n):
    '''Implement n+1-bit toffoli using the approach in Elementary gates'''
        
    assert ctlnumber>=3 #"This method works only for more than 2 control bits"
        
    from sympy.combinatorics.graycode import GrayCode
    gray_code = list(GrayCode(ctlnumber).generate_gray())
    last_pattern = None
        #angle to construct nth square root of diagonlized pauli x matrix
        #via u3(0,lam_angle,0)
    lam_angle = np.pi/(2**(ctlnumber-1))
        #transform to eigenvector basis of pauli X
    qc.h(tgt)
    for pattern in gray_code:
            
        if not '1' in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
            #find left most set bit
        lm_pos = list(pattern).index('1')

            #find changed bit
        comp = [i!=j for i,j in zip(pattern,last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                if ctl[offset+pos] >= n and ctl[offset+lm_pos] >= n:
                    qc.cx(qr2[ctl[offset+pos]-n], qr2[ctl[offset+lm_pos]-n])
                elif ctl[offset+pos]<n and ctl[offset+lm_pos] <n:
                    qc.cx(qr1[ctl[offset+pos]], qr1[ctl[offset+lm_pos]])
                elif ctl[offset+pos]<n and ctl[offset+lm_pos] >= n:
                    qc.cx(qr1[ctl[offset+pos]], qr2[ctl[offset+lm_pos]-n])
                else:
                    qc.cx(qr2[ctl[offset+pos]-n], qr1[ctl[offset+lm_pos]])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    if ctl[offset+idx] >= n and ctl[offset+lm_pos] >= n:
                        qc.cx(qr2[ctl[offset+idx]-n], qr2[ctl[offset+lm_pos]-n])
                    elif ctl[offset+idx]<n and ctl[offset+lm_pos] <n:
                        qc.cx(qr1[ctl[offset+idx]], qr1[ctl[offset+lm_pos]])
                    elif ctl[offset+idx]<n and ctl[offset+lm_pos] >= n:
                        qc.cx(qr1[ctl[offset+idx]], qr2[ctl[offset+lm_pos]-n])
                    else:
                        qc.cx(qr2[ctl[offset+idx]-n], qr1[ctl[offset+lm_pos]])
            #check parity
        if pattern.count('1') % 2 == 0:
                #inverse
            if ctl[offset+lm_pos] < n:
                qc.cu3(0,-lam_angle,0,qr1[ctl[offset+lm_pos]],tgt)
            else:
                qc.cu3(0,-lam_angle,0,qr2[ctl[offset+lm_pos]-n],tgt)
        else:
            if ctl[offset+lm_pos] < n:
                qc.cu3(0,lam_angle,0,qr1[ctl[offset+lm_pos]],tgt)
            else:
                qc.cu3(0,lam_angle,0,qr2[ctl[offset+lm_pos]-n],tgt)
        last_pattern = pattern
    qc.h(tgt)

def read_make_circuit(n, qr1, qr2, meas1, meas2, qc):
    with open("intdiv-esop0-rec{}.txt" .format(n), "r") as f:
        data = f.readlines()

    for i in range(len(data)):
        ctl = []
        xgate = []
        doubledigit = False
        sign1 = False
        ctlnumber = int(data[i][0])-1
        if data[i][1].isdigit():
            ctlnumber = int(data[i][0] + data[i][1])
        if int(data[i][-2]) >= n:
            tgt = qr2[int(data[i][-2]) - n]
        else:
            tgt = qr1[int(data[i][-2])]
        if data[i][-3].isdigit():
            if int(data[i][-3] + data[i][-2]) >= n:
                tgt = qr2[int(data[i][-3] + data[i][-2]) - n]
            else:
                tgt = qr1[int(data[i][-3] + data[i][-2])]           
        for j in range(len(data[i])):
            if data[i][j] == str(" "):
                doubledigit = False
            elif data[i][j] == str("-"):
                sign1 = True
            elif data[i][j].isdigit() and not data[i][j+1].isdigit() and not doubledigit:
                ctl.append(int(data[i][j]))
                if sign1:
                    xgate.append(int(data[i][j]))
                    if int(data[i][j]) >= n:
                        qc.x(qr2[int(data[i][j]) - n])
                    else:
                        qc.x(qr1[int(data[i][j])])
                    sign1 = False
            elif data[i][j].isdigit() and data[i][j+1].isdigit():
                ctl.append(int(data[i][j] + data[i][j+1]))
                doubledigit = True
                if sign1:
                    xgate.append(int(data[i][j] + data[i][j+1]))
                    if int(data[i][j]+ data[i][j+1]) >= n:
                        qc.x(qr2[int(data[i][j] + data[i][j+1]) - n])
                    else:
                        qc.x(qr1[int(data[i][j] + data[i][j+1])])               
                    sign1 = False
        ctl.pop(0)
        ctl.pop(-1)
        print("i ", i)
        print("ctl = ", ctl)
        if ctlnumber == 0: #single not gate
            qc.x(tgt)
        elif ctlnumber == 1: #cnot gate
            if ctl[0] >= n:
                qc.cx(qr2[ctl[0] - n], tgt)
            else:
                qc.cx(qr1[ctl[0]], tgt)
        elif ctlnumber == 2: #toffoli gate
            if ctl[0] >= n and ctl[1] >= n:
                qc.ccx(qr2[ctl[0]-n], qr2[ctl[1]-n], tgt)
            elif ctl[0]<n and ctl[1] <n:
                qc.ccx(qr1[ctl[0]], qr1[ctl[1]], tgt)
            elif ctl[0]<n and ctl[1] >= n:
                qc.ccx(qr1[ctl[0]], qr2[ctl[1]-n], tgt)
            else:
                qc.ccx(qr2[ctl[0]-n], qr1[ctl[1]], tgt)
        else: #not gates controlled with more than 2 qubits
            nc_toffoli(qc, qr1, qr2, ctl, tgt, ctlnumber, 0, n)
        for j in xgate:
            if j >= n:
                qc.x(qr2[j-n])
            else:
                qc.x(qr1[j])

def measure(qc, n, qr1, qr2, meas1, meas2):
    qc.measure(qr1, meas1)
    qc.measure(qr2, meas2)
    job = execute(qc, backend='local_qasm_simulator')
    rd = job.result().get_counts()
    element = [k for k in rd]
    print(element[0])
    real = sum([2**(i) for i, e in enumerate(reversed(element[0])) if e == "1" and i < n])
    inverse = sum([2**-(i+1) for i, e in enumerate(element[0]) if e == "1" and i < n])
    print("Input number = ", real)
    print("Reciprocal number = ", inverse)
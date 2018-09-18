from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt
import numpy as np
#import nc_toffoli
import math

def nc_toffoli(qc, qr, ctl,tgt,n,offset):
    '''Implement n+1-bit toffoli using the approach in Elementary gates'''
        
    assert n>=3 #"This method works only for more than 2 control bits"
        
    from sympy.combinatorics.graycode import GrayCode
    gray_code = list(GrayCode(n).generate_gray())
    last_pattern = None
        #angle to construct nth square root of diagonlized pauli x matrix
        #via u3(0,lam_angle,0)
    lam_angle = np.pi/(2**(n-1))
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
                #print(ctl[offset+idx],ctl[offset+lm_pos])
                qc.cx(qr[ctl[offset+pos]],qr[ctl[offset+lm_pos]])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    #print(ctl[offset+idx],ctl[offset+lm_pos])
                    qc.cx(qr[ctl[offset+idx]],qr[ctl[offset+lm_pos]])
            #check parity
        if pattern.count('1') % 2 == 0:
                #inverse
            #print((offset+lm_pos))
            qc.cu3(0,-lam_angle,0,qr[ctl[offset+lm_pos]],tgt)
        else:
            #print((offset+lm_pos))
            qc.cu3(0,lam_angle,0,qr[ctl[offset+lm_pos]],tgt)
        last_pattern = pattern
    qc.h(tgt)

n = 6
with open("intdiv-esop0-rec{}.txt" .format(n), "r") as f:
#with open("testdouble.txt", "r") as f:
    data = f.readlines()


print(len(data))
#data = np.array(data, dtype=int)
qr = QuantumRegister(2*n)
meas = ClassicalRegister(2*n)
qc = QuantumCircuit(qr, meas)
qc.x(qr[1])
qc.x(qr[3])
for i in range(len(data)):
    ctl = []
    xgate = []
    doubledigit = False
    sign1 = False
    ctlnumber = int(data[i][0])-1
    #print(ctlnumber)
    if data[i][1].isdigit():
        ctlnumber = int(data[i][0] + data[i][1])
    
    tgt = qr[int(data[i][-2])]
    if data[i][-3].isdigit():
        tgt = qr[int(data[i][-3] + data[i][-2])]

    for j in range(len(data[i])):
        if data[i][j] == str(" "):
            doubledigit = False
        elif data[i][j] == str("-"):
            sign1 = True
            #qc.x(qr[int(data[i][j+1])])
        elif data[i][j].isdigit() and not data[i][j+1].isdigit() and not doubledigit:
            ctl.append(int(data[i][j]))
            if sign1:
                xgate.append(int(data[i][j]))
                qc.x(qr[int(data[i][j])])
                sign1 = False
        elif data[i][j].isdigit() and data[i][j+1].isdigit():
            ctl.append(int(data[i][j] + data[i][j+1]))
            doubledigit = True
            if sign1:
                xgate.append(int(data[i][j] + data[i][j+1]))               
                qc.x(qr[int(data[i][j] + data[i][j+1])])
                sign1 = False
    ctl.pop(0)
    ctl.pop(-1)
    print("i ", i)
    print("ctl = ", ctl)
    if ctlnumber == 0: #single not gate
        qc.x(tgt)
    elif ctlnumber == 1: #cnot gate
        qc.cx(qr[ctl[0]], tgt)
    elif ctlnumber == 2: #toffoli gate
        qc.ccx(qr[ctl[0]], qr[ctl[1]], tgt)
    else: #not gates controlled with more than 2 qubits
        #for j in range(1,int(len(data[i])/2)-1):
        #    if len(minus) == 0:
        #        ctl.append(qr[int(data[i][2*j])])
        #        print(int(data[i][2*j]))
            #elif j < minus[0]:
            #    ctl.append(int(data[i][2*j]))
        nc_toffoli(qc, qr, ctl, tgt, ctlnumber, 0)
    for j in xgate:
        qc.x(qr[j])
            
qc.measure(qr, meas)


#qc.measure(qr, meas)
job = execute(qc, backend='local_qasm_simulator')
#counts = job.result().get_counts()
print(job.result().get_counts())
#for j in range(n):
rd = job.result().get_counts()
element = [k for k in rd]
#for j in range(n):
#    print(element[n])
print(element)

real = sum([2**(i) for i, e in enumerate(reversed(element[0])) if e == "1" and i < n])
inverse = sum([2**-(i+1) for i, e in enumerate(element[0]) if e == "1" and i < n])
print(real)
print(inverse)
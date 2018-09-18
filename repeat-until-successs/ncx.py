import ncu1
from qiskit import QuantumCircuit, QuantumRegister
from numpy import angle, pi

def ncx(self, ctls, tgt, global_phase=0):
    """Apply n-controlled X gate from ctls to tgt."""
    
    if isinstance(tgt, (list, QuantumRegister)):
        tgt = tgt[0]

    n = len(ctls)
    if n == 0:
        self.x(tgt)
        if global_phase:
            self.u1(angle(global_phase), tgt)
            self.x(tgt)
            self.u1(angle(global_phase), tgt)
            self.x(tgt)
        return
    if n == 1:
        self.cx(ctls[0], tgt)
        if global_phase:
            self.u1(angle(global_phase), ctls[0])
        return
    if n == 2:
        self.ccx(ctls[0], ctls[1], tgt)
        if global_phase:
            self.cu1(angle(global_phase), ctls[0], ctls[1])
        return

    self.h(tgt)
    self.ncu1(pi, ctls, tgt, global_phase=global_phase)
    self.h(tgt)

QuantumCircuit.ncx = ncx

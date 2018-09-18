from sympy.combinatorics.graycode import GrayCode
from qiskit import QuantumCircuit, QuantumRegister
from numpy import angle

def ncu1(self, lam, ctls, tgt, global_phase=0):
    """Apply n-controlled u1 gate from ctls to tgt with angle lam."""
    
    if isinstance(tgt, (list, QuantumRegister)):
        tgt = tgt[0]

    n = len(ctls)
    if n == 0:
        self.u1(lam, tgt)
        return
    if n == 1:
        self.cu1(lam, ctls[0], tgt)
        return 
    gray_code = list(GrayCode(n).generate_gray())
    last_pattern = None

    lam_angle = lam*(1/(2**(n-1)))
    gp_angle = angle(global_phase)*(1/(2**(n-1)))

    for pattern in gray_code:
        if not '1' in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        #find left most set bit
        lm_pos = list(pattern).index('1')

        #find changed bit
        comp = [i != j for i, j in zip(pattern,last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                self.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    self.cx(ctls[idx], ctls[lm_pos])
        #check parity
        if pattern.count('1') % 2 == 0:
            #inverse
            self.cu1(-lam_angle, ctls[lm_pos], tgt)
            if global_phase:
                self.u1(-gp_angle, ctls[lm_pos])
        else:
            self.cu1(lam_angle, ctls[lm_pos], tgt)
            if global_phase:
                self.u1(gp_angle, ctls[lm_pos])
        last_pattern = pattern

QuantumCircuit.ncu1 = ncu1

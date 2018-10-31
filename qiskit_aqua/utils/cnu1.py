"""
CNU1 gate. N Controlled-U1 Gate. Not Using ancilla qubits.
"""

from sympy.combinatorics.graycode import GrayCode
from qiskit import QuantumCircuit, QuantumRegister, CompositeGate
from numpy import angle


class CNU1Gate(CompositeGate):
    """CNU1 gate."""

    def __init__(self, theta, ctls, tgt, circ=None):
        """Create new CNU1 gate."""
        qubits = [v for v in ctls] + [tgt]
        n_c = len(ctls)
        super(CNU1Gate, self).__init__("cnu1", (theta, n_c), qubits, circ)

        if n_c == 1: # cx
            self.cu1(theta, ctls[0], tgt)
        else:
            self.apply_cnu1(theta, ctls, tgt)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[1]]]
        tgt_bits = self.arg[-1]
        theta = self.param[0]
        self._modifiers(circ.cnu1(theta, ctl_bits, tgt_bits))

    def apply_cnu1(self, theta, ctls, tgt, global_phase=0):
        """Apply n-controlled u1 gate from ctls to tgt with angle theta."""
        
        n = len(ctls)

        gray_code = list(GrayCode(n).generate_gray())
        last_pattern = None

        theta_angle = theta*(1/(2**(n-1)))
        gp_angle = angle(global_phase)*(1/(2**(n-1)))

        for pattern in gray_code:
            if not '1' in pattern:
                continue
            if last_pattern is None:
                last_pattern = pattern
            #find left most set bit
            lm_pos = list(pattern).index('1')

            #find changed bit
            comp = [i != j for i, j in zip(pattern, last_pattern)]
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
                self.cu1(-theta_angle, ctls[lm_pos], tgt)
                if global_phase:
                    self.u1(-gp_angle, ctls[lm_pos])
            else:
                self.cu1(theta_angle, ctls[lm_pos], tgt)
                if global_phase:
                    self.u1(gp_angle, ctls[lm_pos])
            last_pattern = pattern


def cnu1(self, theta, control_qubits, target_qubit):
    """Apply CNU1 to circuit."""
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []
    for qubit in control_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(target_qubit)
    temp.append(target_qubit)
    self._check_dups(temp)
    return self._attach(CNU1Gate(theta, control_qubits, target_qubit, self))


QuantumCircuit.cnu1 = cnu1
CompositeGate.cnu1 = cnu1

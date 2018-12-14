""" 
N Controlled Not Gate using no ancilla qubits,
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CompositeGate
from qiskit_aqua.utils import cnu1
from numpy import pi


class CNXGate(CompositeGate):
    """CNX gate."""

    def __init__(self, ctls, tgt, circ=None):
        """Create new CNX gate."""
        qubits = [v for v in ctls] + [tgt]
        n_c = len(ctls)
        super(CNXGate, self).__init__("cnx", [n_c], qubits, circ)

        if n_c == 1: # cx
            self.cx(ctls[0], tgt)
        elif n_c == 2: # ccx
            self.ccx(ctls[0], ctls[1], tgt)
        else:
            self.h(tgt)
            self.cnu1(pi, ctls, tgt)
            self.h(tgt)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[0]]]
        tgt_bits = self.arg[-1]
        self._modifiers(circ.cnx_na(ctl_bits, tgt_bits))


def cnx_na(self, control_qubits, target_qubit):
    """Apply N Controlled X gate from ctls to tgt."""
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []
    for qubit in control_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(target_qubit)
    temp.append(target_qubit)
    self._check_dups(temp)
    return self._attach(CNXGate(control_qubits, target_qubit, self))
    

QuantumCircuit.cnx_na = cnx_na

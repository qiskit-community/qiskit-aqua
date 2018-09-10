# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
CNX gate. N Controlled-Not Gate.
"""
from qiskit import QuantumCircuit, CompositeGate


class CNXGate(CompositeGate):
    """CNX gate."""

    def __init__(self, ctls, ancis, tgt, circ=None):
        """Create new CNX gate."""
        qubits = [v for v in ctls] + [v for v in ancis] + [tgt]
        n_c = len(ctls)
        n_a = len(ancis)
        super(CNXGate, self).__init__("cnx", (n_c, n_a), qubits, circ)

        if n_c == 1: # cx
            self.cx(ctls[0], tgt)
        elif n_c == 2: # ccx
            self.ccx(ctls[0], ctls[1], tgt)
        else:
            anci_idx = 0
            self.ccx(ctls[0], ctls[1], ancis[anci_idx])
            for idx in range(2, len(ctls) - 1):
                assert anci_idx + 1 < n_a, "length of ancillary qubits are not enough, please use a large one."
                self.ccx(ctls[idx], ancis[anci_idx], ancis[anci_idx+1])
                anci_idx += 1
            self.ccx(ctls[len(ctls)-1], ancis[anci_idx], tgt)
            for idx in (range(2, len(ctls) - 1))[::-1]:
                self.ccx(ctls[idx], ancis[anci_idx-1], ancis[anci_idx])
                anci_idx -= 1
            self.ccx(ctls[0], ctls[1], ancis[anci_idx])

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[0]]]
        anc_bits = [x for x in self.arg[self.param[0]:self.param[0]+self.param[1]]]
        tgt_bits = self.arg[-1]
        self._modifiers(circ.cnx(ctl_bits, anc_bits, tgt_bits))


def cnx(self, control_qubits, ancillary_qubits, target_qubit):
    """Apply CNX to circuit."""
    temp = []
    if ancillary_qubits:
        all_qubits = control_qubits + ancillary_qubits
    else:
        all_qubits = control_qubits
    for qubit in all_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(target_qubit)
    temp.append(target_qubit)
    self._check_dups(temp)
    return self._attach(CNXGate(control_qubits, ancillary_qubits, target_qubit, self))


QuantumCircuit.cnx = cnx
CompositeGate.cnx = cnx

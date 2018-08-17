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

from math import pi, ceil
from qiskit import QuantumCircuit, CompositeGate


class CNXGate(CompositeGate):
    """CNX gate."""

    def __init__(self, q_controls, q_ancilla, q_target, circ=None):
        """Create new CNX gate."""
        qubits = [v for v in q_controls] + [v for v in q_ancilla] + [q_target]
        n_c = len(q_controls)
        n_a = len(q_ancilla)
        super(CNXGate, self).__init__("cnx", (n_c, n_a), qubits, circ)
        self.multicx([*q_controls, q_target], q_ancilla[0] if q_ancilla else None)

    def cccx(self, qrs, angle=pi/4):
        """
            a 3-qubit controlled-NOT.
            An implementation based on Page 17 of Barenco et al.
            Parameters:
                qrs:
                    list of quantum registers. The last qubit is the target, the rest are controls

                angle:
                    default pi/4 when x is not gate
                    set to pi/8 for square root of not
        """
        assert len(qrs) == 4, "There must be exactly 4 qubits of quantum registers for cccx"

        # controlled-V
        self.ch(qrs[0], qrs[3])
        self.cu1(-angle, qrs[0], qrs[3])
        self.ch(qrs[0], qrs[3])
        # ------------

        self.cx(qrs[0], qrs[1])

        # controlled-Vdag
        self.ch(qrs[1], qrs[3])
        self.cu1(angle, qrs[1], qrs[3])
        self.ch(qrs[1], qrs[3])
        # ---------------

        self.cx(qrs[0], qrs[1])

        # controlled-V
        self.ch(qrs[1], qrs[3])
        self.cu1(-angle, qrs[1], qrs[3])
        self.ch(qrs[1], qrs[3])
        # ------------

        self.cx(qrs[1], qrs[2])

        # controlled-Vdag
        self.ch(qrs[2], qrs[3])
        self.cu1(angle, qrs[2], qrs[3])
        self.ch(qrs[2], qrs[3])
        # ---------------

        self.cx(qrs[0], qrs[2])

        # controlled-V
        self.ch(qrs[2], qrs[3])
        self.cu1(-angle, qrs[2], qrs[3])
        self.ch(qrs[2], qrs[3])
        # ------------

        self.cx(qrs[1], qrs[2])

        # controlled-Vdag
        self.ch(qrs[2], qrs[3])
        self.cu1(angle, qrs[2], qrs[3])
        self.ch(qrs[2], qrs[3])
        # ---------------

        self.cx(qrs[0], qrs[2])

        # controlled-V
        self.ch(qrs[2], qrs[3])
        self.cu1(-angle, qrs[2], qrs[3])
        self.ch(qrs[2], qrs[3])

    def ccccx(self, qrs):
        """
           a 4-qubit controlled-NOT.
            An implementation based on Page 21 (Lemma 7.5) of Barenco et al.
            Parameters:
                qrs:
                    list of quantum registers. The last qubit is the target, the rest are controls
        """
        assert len(qrs) == 5, "There must be exactly 5 qubits for ccccx"

        # controlled-V
        self.ch(qrs[3], qrs[4])
        self.cu1(-pi / 2, qrs[3], qrs[4])
        self.ch(qrs[3], qrs[4])
        # ------------

        self.cccx(qrs[:4])

        # controlled-Vdag
        self.ch(qrs[3], qrs[4])
        self.cu1(pi / 2, qrs[3], qrs[4])
        self.ch(qrs[3], qrs[4])
        # ------------

        self.cccx(qrs[:4])

        self.cccx([qrs[0], qrs[1], qrs[2], qrs[4]], angle=pi / 8)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[0]]]
        anc_bits = [x for x in self.arg[self.param[0]:self.param[0]+self.param[1]]]
        tgt_bits = self.arg[-1]
        self._modifiers(circ.cnx(ctl_bits, anc_bits, tgt_bits))

    def multicx(self, qrs, qancilla=None):
        """
            construct a circuit for multi-qubit controlled not
            Parameters:
                self:
                    quantum circuit
                qrs:
                    list of quantum registers of at least length 1
                qancilla:
                    a quantum register. can be None if len(qrs) <= 5

            Returns:
                qc:
                    a circuit appended with multi-qubit cnot
        """
        if len(qrs) <= 0:
            pass
        elif len(qrs) == 1:
            self.x(qrs[0])
        elif len(qrs) == 2:
            self.cx(qrs[0], qrs[1])
        elif len(qrs) == 3:
            self.ccx(qrs[0], qrs[1], qrs[2])
        elif len(qrs) == 4:
            self.cccx(qrs)
        elif len(qrs) == 5:
            self.ccccx(qrs)
        else:  # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target, and qancilla as working qubit
            assert qancilla is not None, "There must be an ancilla qubit not necesseraly initialized to zero"
            n = len(qrs) + 1  # SOME ERROR HERE
            m1 = ceil(n / 2)
            m2 = n - m1 - 1

            self.multicx([*qrs[:m1], qancilla], qrs[m1])

            self.multicx([*qrs[m1:m1 + m2 - 1], qancilla, qrs[n - 2]], qrs[m1 - 1])

            self.multicx([*qrs[:m1], qancilla], qrs[m1])

            self.multicx([*qrs[m1:m1 + m2 - 1], qancilla, qrs[n - 2]], qrs[m1 - 1])


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

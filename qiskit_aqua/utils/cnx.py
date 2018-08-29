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
from qiskit import QuantumCircuit, CompositeGate, QuantumRegister


class CNXGate(CompositeGate):
    """CNX gate."""

    def __init__(self, control_qubits, target_qubit, ancillary_qubits, circ=None, mode='basic'):
        """Create new CNX gate."""
        self._mode = mode
        qubits = control_qubits + ancillary_qubits + [target_qubit]
        super(CNXGate, self).__init__("cnx", (len(control_qubits), len(ancillary_qubits)), qubits, circ)
        if self._mode == 'basic':
            self.ccx_v_chain(control_qubits, target_qubit, ancillary_qubits)
        elif self._mode == 'advanced':
            self.multicx([*control_qubits, target_qubit], ancillary_qubits[0] if ancillary_qubits else None)
        else:
            raise ValueError('Unrecognized mode for building cnx gate: {}.'.format(self._mode))

    def ccx_v_chain(self, control_qubits, target_qubit, ancillary_qubits):
        """Create new CNX gate by chaining ccx gates into a V shape."""
        anci_idx = 0
        self.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
        for idx in range(2, len(control_qubits) - 1):
            assert anci_idx + 1 < len(ancillary_qubits), "Insufficient number of ancillary qubits."
            self.ccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
            anci_idx += 1
        self.ccx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx], target_qubit)
        for idx in (range(2, len(control_qubits) - 1))[::-1]:
            self.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
            anci_idx -= 1
        self.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])

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

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[0]]]
        anc_bits = [x for x in self.arg[self.param[0]:self.param[0]+self.param[1]]]
        tgt_bits = self.arg[-1]
        self._modifiers(circ.cnx(ctl_bits, tgt_bits, anc_bits, mode=self._mode))


def cnx(self, q_controls, q_target, q_ancilla, mode='basic'):
    """Apply CNX to circuit."""
    if len(q_controls) == 1:  # cx
        self.cx(q_controls[0], q_target)
    elif len(q_controls) == 2:  # ccx
        self.ccx(q_controls[0], q_controls[1], q_target)
    else:

        # check controls
        if isinstance(q_controls, QuantumRegister):
            control_qubits = [qb for qb in q_controls]
        elif isinstance(q_controls, list):
            control_qubits = q_controls
        else:
            raise ValueError('CNX gate needs a list of qubits or a quantum register for controls.')

        # check target
        if isinstance(q_target, tuple):
            target_qubit = q_target
        else:
            raise ValueError('CNX gate needs a single qubit as target.')

        # check ancilla
        if q_ancilla is None:
            ancillary_qubits = []
        elif isinstance(q_ancilla, QuantumRegister):
            ancillary_qubits = [qb for qb in q_ancilla]
        elif isinstance(q_ancilla, list):
            ancillary_qubits = q_ancilla
        else:
            raise ValueError('CNX gate needs None or a list of qubits or a quantum register for ancilla.')

        all_qubits = control_qubits + [target_qubit] + ancillary_qubits
        for qubit in all_qubits:
            self._check_qubit(qubit)
        self._check_dups(all_qubits)
        return self._attach(CNXGate(control_qubits, target_qubit, ancillary_qubits, self, mode=mode))


QuantumCircuit.cnx = cnx
CompositeGate.cnx = cnx

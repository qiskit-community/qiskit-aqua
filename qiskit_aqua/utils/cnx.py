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
Multiple-(N)-Controlled-Not Operation.
"""

from math import pi, ceil
from qiskit import QuantumCircuit, QuantumRegister


def _ccx_v_chain(qc, control_qubits, target_qubit, ancillary_qubits):
    """Create new CNX circuit by chaining ccx gates into a V shape."""
    anci_idx = 0
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
    for idx in range(2, len(control_qubits) - 1):
        assert anci_idx + 1 < len(ancillary_qubits), "Insufficient number of ancillary qubits."
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
        anci_idx += 1
    qc.ccx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx], target_qubit)
    for idx in (range(2, len(control_qubits) - 1))[::-1]:
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
        anci_idx -= 1
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])


def _cccx(qc, qrs, angle=pi / 4):
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
    qc.ch(qrs[0], qrs[3])
    qc.cu1(-angle, qrs[0], qrs[3])
    qc.ch(qrs[0], qrs[3])
    # ------------

    qc.cx(qrs[0], qrs[1])

    # controlled-Vdag
    qc.ch(qrs[1], qrs[3])
    qc.cu1(angle, qrs[1], qrs[3])
    qc.ch(qrs[1], qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[1])

    # controlled-V
    qc.ch(qrs[1], qrs[3])
    qc.cu1(-angle, qrs[1], qrs[3])
    qc.ch(qrs[1], qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.ch(qrs[2], qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.ch(qrs[2], qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.ch(qrs[2], qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.ch(qrs[2], qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.ch(qrs[2], qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.ch(qrs[2], qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.ch(qrs[2], qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.ch(qrs[2], qrs[3])


def _ccccx(qc, qrs):
    """
       a 4-qubit controlled-NOT.
        An implementation based on Page 21 (Lemma 7.5) of Barenco et al.
        Parameters:
            qrs:
                list of quantum registers. The last qubit is the target, the rest are controls
    """
    assert len(qrs) == 5, "There must be exactly 5 qubits for ccccx"

    # controlled-V
    qc.ch(qrs[3], qrs[4])
    qc.cu1(-pi / 2, qrs[3], qrs[4])
    qc.ch(qrs[3], qrs[4])
    # ------------

    _cccx(qc, qrs[:4])

    # controlled-Vdag
    qc.ch(qrs[3], qrs[4])
    qc.cu1(pi / 2, qrs[3], qrs[4])
    qc.ch(qrs[3], qrs[4])
    # ------------

    _cccx(qc, qrs[:4])
    _cccx(qc, [qrs[0], qrs[1], qrs[2], qrs[4]], angle=pi / 8)


def _multicx(qc, qrs, qancilla=None):
    """
        construct a circuit for multi-qubit controlled not
        Parameters:
            qc:
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
        qc.x(qrs[0])
    elif len(qrs) == 2:
        qc.cx(qrs[0], qrs[1])
    elif len(qrs) == 3:
        qc.ccx(qrs[0], qrs[1], qrs[2])
    elif len(qrs) == 4:
        _cccx(qc, qrs)
    elif len(qrs) == 5:
        _ccccx(qc, qrs)
    else:  # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target, and qancilla as working qubit
        assert qancilla is not None, "There must be an ancilla qubit not necesseraly initialized to zero"
        n = len(qrs) + 1  # SOME ERROR HERE
        m1 = ceil(n / 2)
        m2 = n - m1 - 1
        _multicx(qc, [*qrs[:m1], qancilla], qrs[m1])
        _multicx(qc, [*qrs[m1:m1 + m2 - 1], qancilla, qrs[n - 2]], qrs[m1 - 1])
        _multicx(qc, [*qrs[:m1], qancilla], qrs[m1])
        _multicx(qc, [*qrs[m1:m1 + m2 - 1], qancilla, qrs[n - 2]], qrs[m1 - 1])


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
            raise ValueError('CNX needs a list of qubits or a quantum register for controls.')

        # check target
        if isinstance(q_target, tuple):
            target_qubit = q_target
        else:
            raise ValueError('CNX needs a single qubit as target.')

        # check ancilla
        if q_ancilla is None:
            ancillary_qubits = []
        elif isinstance(q_ancilla, QuantumRegister):
            ancillary_qubits = [qb for qb in q_ancilla]
        elif isinstance(q_ancilla, list):
            ancillary_qubits = q_ancilla
        else:
            raise ValueError('CNX needs None or a list of qubits or a quantum register for ancilla.')

        all_qubits = control_qubits + [target_qubit] + ancillary_qubits
        for qubit in all_qubits:
            self._check_qubit(qubit)
        self._check_dups(all_qubits)

        if mode == 'basic':
            _ccx_v_chain(self, control_qubits, target_qubit, ancillary_qubits)
        elif mode == 'advanced':
            _multicx(self, [*control_qubits, target_qubit], ancillary_qubits[0] if ancillary_qubits else None)
        else:
            raise ValueError('Unrecognized mode for building CNX circuit: {}.'.format(mode))


QuantumCircuit.cnx = cnx

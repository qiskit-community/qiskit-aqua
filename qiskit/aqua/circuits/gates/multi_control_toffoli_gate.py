# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Multiple-Control Toffoli Gate.
"""

import logging
from math import pi, ceil

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import AquaError
from qiskit.aqua.utils.circuit_utils import is_qubit

logger = logging.getLogger(__name__)


def _ccx_v_chain(qc, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=False):
    """
    Create new MCT circuit by chaining ccx gates into a V shape.

    The dirty_ancilla mode is from https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2
    """

    if len(ancillary_qubits) < len(control_qubits) - 2:
        raise AquaError('Insufficient number of ancillary qubits.')

    if dirty_ancilla:
        anci_idx = len(control_qubits) - 3
        qc.ccx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx], target_qubit)
        for idx in reversed(range(2, len(control_qubits) - 1)):
            qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
            anci_idx -= 1

    anci_idx = 0
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
    for idx in range(2, len(control_qubits) - 1):
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
        anci_idx += 1
    qc.ccx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx], target_qubit)
    for idx in reversed(range(2, len(control_qubits) - 1)):
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
        anci_idx -= 1
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])

    if dirty_ancilla:
        anci_idx = 0
        for idx in range(2, len(control_qubits) - 1):
            qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
            anci_idx += 1


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
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[0], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[0], qrs[1])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[1], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[1])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[1], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.h(qrs[3])


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
    qc.h(qrs[4])
    qc.cu1(-pi / 2, qrs[3], qrs[4])
    qc.h(qrs[4])
    # ------------

    _cccx(qc, qrs[:4])

    # controlled-Vdag
    qc.h(qrs[4])
    qc.cu1(pi / 2, qrs[3], qrs[4])
    qc.h(qrs[4])
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
    else:
        _multicx_recursion(qc, qrs, qancilla)


def _multicx_recursion(qc, qrs, qancilla=None):
    if len(qrs) == 4:
        _cccx(qc, qrs)
    elif len(qrs) == 5:
        _ccccx(qc, qrs)
    else:  # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target, and qancilla as working qubit
        assert qancilla is not None, "There must be an ancilla qubit not necesseraly initialized to zero"
        n = len(qrs)
        m1 = ceil(n / 2)
        _multicx_recursion(qc, [*qrs[:m1], qancilla], qrs[m1])
        _multicx_recursion(qc, [*qrs[m1:n - 1], qancilla, qrs[n - 1]], qrs[m1 - 1])
        _multicx_recursion(qc, [*qrs[:m1], qancilla], qrs[m1])
        _multicx_recursion(qc, [*qrs[m1:n - 1], qancilla, qrs[n - 1]], qrs[m1 - 1])


def _multicx_noancilla(qc, qrs):
    """
        construct a circuit for multi-qubit controlled not without ancillary
        qubits
        Parameters:
            qc:
                quantum circuit
            qrs:
                list of quantum registers of at least length 1

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
    else:
        # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target
        ctls = qrs[:-1]
        tgt = qrs[-1]
        qc.h(tgt)
        qc.mcu1(pi, ctls, tgt)
        qc.h(tgt)


def mct(self, q_controls, q_target, q_ancilla, mode='basic'):
    """
    Apply Multiple-Control Toffoli operation
    Args:
        q_controls: The list of control qubits
        q_target: The target qubit
        q_ancilla: The list of ancillary qubits
        mode (string): The implementation mode to use
    """

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
            raise AquaError('MCT needs a list of qubits or a quantum register for controls.')

        # check target
        if is_qubit(q_target):
            target_qubit = q_target
        else:
            raise AquaError('MCT needs a single qubit as target.')

        # check ancilla
        if q_ancilla is None:
            ancillary_qubits = []
        elif isinstance(q_ancilla, QuantumRegister):
            ancillary_qubits = [qb for qb in q_ancilla]
        elif isinstance(q_ancilla, list):
            ancillary_qubits = q_ancilla
        else:
            raise AquaError('MCT needs None or a list of qubits or a quantum register for ancilla.')

        all_qubits = control_qubits + [target_qubit] + ancillary_qubits

        self._check_qargs(all_qubits)
        self._check_dups(all_qubits)

        if mode == 'basic':
            _ccx_v_chain(self, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=False)
        elif mode == 'basic-dirty-ancilla':
            _ccx_v_chain(self, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=True)
        elif mode == 'advanced':
            _multicx(self, [*control_qubits, target_qubit], ancillary_qubits[0] if ancillary_qubits else None)
        elif mode == 'noancilla':
            _multicx_noancilla(self, [*control_qubits, target_qubit])
        else:
            raise AquaError('Unrecognized mode for building MCT circuit: {}.'.format(mode))


def cnx(self, *args, **kwargs):
    logger.warning("The gate name 'cnx' will be deprecated. Please use 'mct' (Multiple-Control Toffoli) instead.")
    return mct(self, *args, **kwargs)


QuantumCircuit.mct = mct
QuantumCircuit.cnx = cnx

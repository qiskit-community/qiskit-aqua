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
Multiple-Control U3 gate. Not using ancillary qubits.
"""

import logging

from sympy.combinatorics.graycode import GrayCode
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua.utils.controlled_circuit import apply_cu3

logger = logging.getLogger(__name__)


def _apply_mcu3(circuit, theta, phi, lam, ctls, tgt):
    """Apply multi-controlled u3 gate from ctls to tgt with angles theta,
    phi, lam."""

    n = len(ctls)

    gray_code = list(GrayCode(n).generate_gray())
    last_pattern = None

    theta_angle = theta*(1/(2**(n-1)))
    phi_angle = phi*(1/(2**(n-1)))
    lam_angle = lam*(1/(2**(n-1)))

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
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        #check parity
        if pattern.count('1') % 2 == 0:
            #inverse
            apply_cu3(circuit, -theta_angle, phi_angle, lam_angle,
                      ctls[lm_pos], tgt)
        else:
            apply_cu3(circuit, theta_angle, phi_angle, lam_angle,
                      ctls[lm_pos], tgt)
        last_pattern = pattern


def mcu3(self, theta, phi, lam, control_qubits, target_qubit):
    """
    Apply Multiple-Controlled U3 gate
    Args:
        theta: angle theta
        phi: angle phi
        lam: angle lambda
        control_qubits: The list of control qubits
        target_qubit: The target qubit
    """
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []

    self._check_qargs(control_qubits)
    temp += control_qubits

    self._check_qargs([target_qubit])
    temp.append(target_qubit)

    self._check_dups(temp)
    n_c = len(control_qubits)
    if n_c == 1:  # cu3
        apply_cu3(self, theta, phi, lam, control_qubits[0], target_qubit)
    else:
        _apply_mcu3(self, theta, phi, lam, control_qubits, target_qubit)


QuantumCircuit.mcu3 = mcu3

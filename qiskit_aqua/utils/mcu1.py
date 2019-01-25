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
Multiple-Control U1 gate. Not using ancillary qubits.
"""

from sympy.combinatorics.graycode import GrayCode
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CompositeGate
from qiskit_aqua.utils.controlledcircuit import apply_cu1
from numpy import angle


class MCU1Gate(CompositeGate):
    """MCU1 gate."""

    def __init__(self, theta, ctls, tgt, circ=None):
        """Create new MCU1 gate."""
        self._ctl_bits = ctls
        self._tgt_bits = tgt
        self._theta = theta
        qubits = [v for v in ctls] + [tgt]
        n_c = len(ctls)
        super(MCU1Gate, self).__init__("mcu1", (theta, n_c), qubits, circ)

        if n_c == 1: # cx
            apply_cu1(circ, theta, ctls[0], tgt)
        else:
            self.apply_mcu1(theta, ctls, tgt, circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.mcu1(self._theta, self._ctl_bits, self._tgt_bits))

    def apply_mcu1(self, theta, ctls, tgt, circuit, global_phase=0):
        """Apply multi-controlled u1 gate from ctls to tgt with angle theta."""

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
                    circuit.cx(ctls[pos], ctls[lm_pos])
                else:
                    indices = [i for i, x in enumerate(pattern) if x == '1']
                    for idx in indices[1:]:
                        circuit.cx(ctls[idx], ctls[lm_pos])
            #check parity
            if pattern.count('1') % 2 == 0:
                #inverse
                apply_cu1(circuit, -theta_angle, ctls[lm_pos], tgt)
                if global_phase:
                    circuit.u1(-gp_angle, ctls[lm_pos])
            else:
                apply_cu1(circuit, theta_angle, ctls[lm_pos], tgt)
                if global_phase:
                    circuit.u1(gp_angle, ctls[lm_pos])
            last_pattern = pattern


def mcu1(self, theta, control_qubits, target_qubit):
    """Apply MCU1 to circuit."""
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []
    for qubit in control_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(target_qubit)
    temp.append(target_qubit)
    self._check_dups(temp)
    return self._attach(MCU1Gate(theta, control_qubits, target_qubit, self))


QuantumCircuit.mcu1 = mcu1

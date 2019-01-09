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
CNU3 gate. N Controlled-U3 Gate. Not Using ancilla qubits.
"""

from sympy.combinatorics.graycode import GrayCode
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CompositeGate
from qiskit_aqua.utils.controlledcircuit import apply_cu3
from numpy import angle


class CNU3Gate(CompositeGate):
    """CNU3 gate."""

    def __init__(self, theta, phi, lam, ctls, tgt, circ=None):
        """Create new CNU3 gate."""
        self._ctl_bits = ctls
        self._tgt_bits = tgt
        self._theta = theta
        self._phi = phi
        self._lambda = lam
        qubits = [v for v in ctls] + [tgt]
        n_c = len(ctls)
        super(CNU3Gate, self).__init__("cnu3", (theta, phi, lam, n_c), qubits, circ)

        if n_c == 1: # cx
            self.cu3(theta, phi, lam, ctls[0], tgt)
        else:
            self.apply_cnu3(theta, phi, lam, ctls, tgt, circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cnu3(self._theta, self._phi, self._lambda,
                                  self._ctl_bits, self._tgt_bits))

    def apply_cnu3(self, theta, phi, lam, ctls, tgt, circuit):
        """Apply n-controlled u1 gate from ctls to tgt with angle theta."""
        
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


def cnu3(self, theta, phi, lam, control_qubits, target_qubit):
    """Apply CNU3 to circuit."""
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []
    for qubit in control_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(target_qubit)
    temp.append(target_qubit)
    self._check_dups(temp)
    return self._attach(CNU3Gate(theta, phi, lam, control_qubits, target_qubit, self))


QuantumCircuit.cnu3 = cnu3

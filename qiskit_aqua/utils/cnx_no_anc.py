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
N Controlled Not Gate using no ancilla qubits,
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CompositeGate
from math import pi


class CNXGate(CompositeGate):
    """CNX gate."""

    def __init__(self, ctls, tgt, circ=None):
        """Create new CNX gate."""
        qubits = [v for v in ctls] + [tgt]
        n_c = len(ctls)
        super(CNXGate, self).__init__("cnx", [n_c], qubits, circ)

        if n_c == 1: # cx
            circ.cx(ctls[0], tgt)
        elif n_c == 2: # ccx
            circ.ccx(ctls[0], ctls[1], tgt)
        else:
            self.apply_cnx_na(ctls, tgt, circ)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        ctl_bits = [x for x in self.arg[:self.param[0]]]
        tgt_bits = self.arg[-1]
        self._modifiers(circ.cnx_na(ctl_bits, tgt_bits))

    def apply_cnx_na(self, ctls, tgt, circuit):
        circuit.h(tgt)
        circuit.cnu1(pi, ctls, tgt)
        circuit.h(tgt)

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

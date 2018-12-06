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
N-Controlled-Not Operation.
"""
from qiskit import QuantumCircuit


def cnx(self, ctls, ancis, tgt):
    """Apply CNX to circuit."""
    temp = []
    if ancis:
        all_qubits = ctls + ancis
    else:
        all_qubits = ctls
    for qubit in all_qubits:
        self._check_qubit(qubit)
        temp.append(qubit)
    self._check_qubit(tgt)
    temp.append(tgt)
    self._check_dups(temp)

    n_c = len(ctls)
    n_a = len(ancis)

    if n_c == 1:  # cx
        self.cx(ctls[0], tgt)
    elif n_c == 2:  # ccx
        self.ccx(ctls[0], ctls[1], tgt)
    else:
        anci_idx = 0
        self.ccx(ctls[0], ctls[1], ancis[anci_idx])
        for idx in range(2, len(ctls) - 1):
            assert anci_idx + 1 < n_a, "Not enough ancillary qubits."
            self.ccx(ctls[idx], ancis[anci_idx], ancis[anci_idx + 1])
            anci_idx += 1
        self.ccx(ctls[len(ctls) - 1], ancis[anci_idx], tgt)
        for idx in (range(2, len(ctls) - 1))[::-1]:
            self.ccx(ctls[idx], ancis[anci_idx - 1], ancis[anci_idx])
            anci_idx -= 1
        self.ccx(ctls[0], ctls[1], ancis[anci_idx])


QuantumCircuit.cnx = cnx

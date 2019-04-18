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

from qiskit.aqua.utils import CircuitFactory


class SPsi0Factory(CircuitFactory):

    def __init__(self, num_target_qubits, i_objective):
        super().__init__(num_target_qubits)
        self.i_objective = i_objective

    def required_ancillas(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return 1

    def required_ancillas_controlled(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return 1

    def build(self, qc, q, q_ancillas=None):
        if self.num_target_qubits == 1:
            qc.z(q[0])
        else:
            qc.x(q_ancillas[0])
            qc.h(q_ancillas[0])
            qc.x(q[self.i_objective])
            qc.cx(q[self.i_objective], q_ancillas[0])
            qc.x(q[self.i_objective])
            qc.h(q_ancillas[0])
            qc.x(q_ancillas[0])

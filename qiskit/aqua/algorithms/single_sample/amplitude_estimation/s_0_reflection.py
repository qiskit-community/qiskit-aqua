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
from qiskit.aqua.circuits.gates import ch
from qiskit.aqua.circuits.gates import mct


class S0Factory(CircuitFactory):

    def __init__(self, num_target_qubits):
        super().__init__(num_target_qubits)

    def required_ancillas(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return max(1, self._num_target_qubits - 1)

    def required_ancillas_controlled(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return self._num_target_qubits

    def build(self, qc, q, q_ancillas=None, params=None):
        if self.num_target_qubits == 1:
            qc.z(q[0])
        else:
            for q_ in q:
                qc.x(q_)
            qc.x(q_ancillas[0])
            qc.h(q_ancillas[0])
            q_controls = [q[i] for i in range(len(q))]
            q_ancillas_ = [q_ancillas[i] for i in range(len(q_ancillas))]
            qc.mct(q_controls, q_ancillas_[0], q_ancillas_[1:])
            qc.h(q_ancillas[0])
            qc.x(q_ancillas[0])
            for q_ in q:
                qc.x(q_)

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        self.build(qc, q, q_ancillas, params)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        if self.num_target_qubits == 1:
            qc.cz(q_control, q[0])
        else:
            for q_ in q:
                qc.cx(q_control, q_)
            qc.cx(q_control, q_ancillas[0])
            qc.ch(q_control, q_ancillas[0])
            q_controls = [q_control] + [q[i] for i in range(len(q))]
            q_ancillas_ = [q_ancillas[i] for i in range(len(q_ancillas))]
            qc.mct(q_controls, q_ancillas_[0], q_ancillas_[1:])
            qc.ch(q_control, q_ancillas[0])
            qc.cx(q_control, q_ancillas[0])
            for q_ in q:
                qc.cx(q_control, q_)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        self.build_controlled(qc, q, q_control, q_ancillas, params)

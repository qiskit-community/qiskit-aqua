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

from qiskit.aqua.algorithms.single_sample.amplitude_estimation.s_psi_0_reflection import SPsi0Factory
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.s_0_reflection import S0Factory


class QFactory(CircuitFactory):

    def __init__(self, a_factory):

        super().__init__(a_factory.num_target_qubits)

        # store A factory
        self.a_factory = a_factory

        # construct reflection factories
        self.s_psi_0_reflection_factory = SPsi0Factory(a_factory.num_target_qubits)
        self.s_0_reflection_factory = S0Factory(a_factory.num_target_qubits)

        # determine number of required ancillas (A does not need to be controlled within Q!)
        self._num_a_ancillas = a_factory.required_ancillas()
        self._num_s_psi_0_ancillas = self.s_psi_0_reflection_factory.required_ancillas()
        self._num_s_psi_0_ancillas_controlled = self.s_psi_0_reflection_factory.required_ancillas_controlled()
        self._num_s_0_ancillas = self.s_0_reflection_factory.required_ancillas()
        self._num_s_0_ancillas_controlled = self.s_0_reflection_factory.required_ancillas_controlled()
        self._num_ancillas = max(self._num_s_psi_0_ancillas, self._num_s_0_ancillas, self._num_a_ancillas)
        self._num_ancillas_controlled = max(self._num_s_psi_0_ancillas_controlled, self._num_s_0_ancillas_controlled, self._num_a_ancillas)

        # determine total number of qubits
        self._num_qubits = a_factory.num_target_qubits + self._num_ancillas
        self._num_qubits_controlled = a_factory.num_target_qubits + self._num_ancillas_controlled

        # get the params
        self._params = a_factory._params

    def required_ancillas(self):
        return self._num_ancillas

    def required_ancillas_controlled(self):
        return self._num_ancillas_controlled

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params
        self.s_psi_0_reflection_factory.build(qc, q, q_ancillas, params)
        self.a_factory.build_inverse(qc, q, q_ancillas, params)
        self.s_0_reflection_factory.build(qc, q, q_ancillas, params)
        self.a_factory.build(qc, q, q_ancillas, params)

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params
        self.a_factory.build_inverse(qc, q, q_ancillas, params)
        self.s_0_reflection_factory.build_inverse(qc, q, q_ancillas, params)
        self.a_factory.build(qc, q, q_ancillas, params)
        self.s_psi_0_reflection_factory.build_inverse(qc, q, q_ancillas, params)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params
        # A operators do not need to be controlled, since they cancel out in the not-controlled case
        self.s_psi_0_reflection_factory.build_controlled(qc, q, q_control, q_ancillas, params)
        self.a_factory.build_inverse(qc, q, q_ancillas, params)
        self.s_0_reflection_factory.build_controlled(qc, q, q_control, q_ancillas, params)
        self.a_factory.build(qc, q, q_ancillas, params)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params
        # A operators do not need to be controlled, since they cancel out in the not-controlled case
        self.a_factory.build_inverse(qc, q, q_ancillas, params)
        self.s_0_reflection_factory.build_controlled_inverse(qc, q, q_control, q_ancillas, params)
        self.a_factory.build(qc, q, q_ancillas, params)
        self.s_psi_0_reflection_factory.build_controlled_inverse(qc, q, q_control, q_ancillas, params)

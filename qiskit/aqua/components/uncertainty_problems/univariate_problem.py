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
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem


class UnivariateProblem(UncertaintyProblem):

    def __init__(self, uncertainty_model, univariate_objective, params=None):

        # determine number of target qubits
        num_target_qubits = uncertainty_model.num_target_qubits + 1
        super().__init__(num_target_qubits)

        # store operators
        self._uncertainty_model = uncertainty_model
        self._univariate_objective = univariate_objective

        # set params
        if params is not None:
            self._params = params
        else:
            self._params = {
                'i_state': range(uncertainty_model.num_target_qubits),
                'i_objective': uncertainty_model.num_target_qubits
            }

    def value_to_estimation(self, value):
        return self._univariate_objective.value_to_estimation(value)

    def required_ancillas(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_objective_ancillas = self._univariate_objective.required_ancillas()
        return max([num_uncertainty_ancillas, num_objective_ancillas])

    def build(self, qc, q, q_ancillas=None, params=None):

        # apply uncertainty model
        self._uncertainty_model.build(qc, q, q_ancillas, params)

        # apply objective function
        self._univariate_objective.build(qc, q, q_ancillas, params)

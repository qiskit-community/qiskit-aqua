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

    """
    Univariate uncertainty problem.
    """

    def __init__(self, uncertainty_model, univariate_objective, i_state=None, i_objective=None):
        """
        Constructor.

        Args:
            uncertainty_model (UnivariateUncertaintyModel): univariate uncertainty model to
            univariate_objective (UnivariatePiecewiseLinearObjective): objective function based on uncertainty
            i_state(int): indices of qubits representing uncertainty
            i_objective: index of qubit representing the objective value in the amplitude
        """

        # determine number of target qubits
        num_target_qubits = uncertainty_model.num_target_qubits + 1
        super().__init__(num_target_qubits)

        # store operators
        self._uncertainty_model = uncertainty_model
        self._univariate_objective = univariate_objective

        # set params
        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        self.i_state = i_state
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits
        self.i_objective = i_objective

    def value_to_estimation(self, value):
        return self._univariate_objective.value_to_estimation(value)

    def required_ancillas(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_objective_ancillas = self._univariate_objective.required_ancillas()
        return max([num_uncertainty_ancillas, num_objective_ancillas])

    def build(self, qc, q, q_ancillas=None):

        q_state = [q[i] for i in self.i_state]
        q_objective = q[self.i_objective]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        # apply objective function
        self._univariate_objective.build(qc, q_state + [q_objective], q_ancillas)

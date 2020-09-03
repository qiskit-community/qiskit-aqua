# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Univariate uncertainty problem. """

from typing import Optional, List
from qiskit.aqua.components.uncertainty_models import UnivariateDistribution
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from .univariate_piecewise_linear_objective import UnivariatePiecewiseLinearObjective


class UnivariateProblem(UncertaintyProblem):

    """
    Univariate uncertainty problem.
    """

    def __init__(self,
                 uncertainty_model: UnivariateDistribution,
                 univariate_objective: UnivariatePiecewiseLinearObjective,
                 i_state: Optional[List[int]] = None,
                 i_objective: Optional[int] = None) -> None:
        """
        Constructor.

        Args:
            uncertainty_model: univariate uncertainty model to
            univariate_objective: objective function based on uncertainty
            i_state: indices of qubits representing uncertainty
            i_objective: index of qubit representing the objective
                            value in the amplitude
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

    def build(self, qc, q, q_ancillas=None, params=None):

        q_state = [q[i] for i in self.i_state]
        q_objective = q[self.i_objective]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        # apply objective function
        self._univariate_objective.build(qc, q_state + [q_objective], q_ancillas)

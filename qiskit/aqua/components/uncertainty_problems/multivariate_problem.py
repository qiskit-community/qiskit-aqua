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

"""
Multivariate Uncertainty Problem.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.components.uncertainty_models import MultivariateDistribution
from .univariate_piecewise_linear_objective import UnivariatePiecewiseLinearObjective


class MultivariateProblem(UncertaintyProblem):

    """
    Multivariate Uncertainty Problem.
    """

    def __init__(self,
                 uncertainty_model: MultivariateDistribution,
                 aggregation_function: CircuitFactory,
                 univariate_objective: UnivariatePiecewiseLinearObjective,
                 conditions:
                 Optional[Union[List[Tuple[int, CircuitFactory]], np.ndarray]] = None) -> None:
        """
        Constructor.

        Args:
            uncertainty_model: multivariate uncertainty model
            aggregation_function: aggregation function that maps
                        the multiple dimension to an aggregated value
            univariate_objective: objective function applied to the aggregated value
            conditions: list of pairs (int, CircuitFactory) =
                target dimension of uncertainty model and condition to be satisfied
                to apply the aggregation
        """

        # determine number of target qubits
        num_target_qubits = uncertainty_model.num_target_qubits + 1
        super().__init__(num_target_qubits)

        # set qubit indices for
        self.i_state = list(range(num_target_qubits - 1))
        self.i_objective = num_target_qubits - 1

        # store parameters
        self._uncertainty_model = uncertainty_model
        self._aggregation_function = aggregation_function
        self._univariate_objective = univariate_objective
        self._conditions = conditions

    def value_to_estimation(self, value):
        if hasattr(self._univariate_objective, 'value_to_estimation'):
            return self._univariate_objective.value_to_estimation(value)
        else:
            return value

    def required_ancillas(self):

        num_condition_ancillas = 0
        num_condition_target_ancillas = 0
        num_aggregation_ancillas = self._aggregation_function.required_ancillas()
        if self._conditions is not None:
            num_condition_target_ancillas = len(self._conditions) + 1 * (len(self._conditions) > 1)
            num_aggregation_ancillas = self._aggregation_function.required_ancillas_controlled()
        if self._conditions is not None:
            for _, condition in self._conditions:
                num_condition_ancillas = np.maximum(num_condition_ancillas,
                                                    condition.required_ancillas())

        # get maximal number of required ancillas
        num_ancillas = max([self._uncertainty_model.required_ancillas(),
                            num_aggregation_ancillas,
                            self._univariate_objective.required_ancillas(),
                            num_condition_ancillas])

        # add ancillas that are required to compute intermediate
        # states are are no directly uncomputed
        num_ancillas += self._aggregation_function.num_sum_qubits
        num_ancillas += num_condition_target_ancillas

        return num_ancillas

    def build(self, qc, q, q_ancillas=None, params=None):

        # apply uncertainty model (can use all ancillas and returns all clean)
        q_state = [q[i] for i in self.i_state]
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        qc.barrier()

        # get all qubits up to the largest state qubit
        num_agg_qubits = self._aggregation_function.num_sum_qubits
        q_agg_in = q_state
        q_agg_out = [q_ancillas[i] for i in range(num_agg_qubits)]
        q_agg = q_agg_in + q_agg_out

        # determine objective qubits (aggregation qubits + objective qubit)
        q_obj = q_agg_out + [q[self.i_objective]]

        # set condition target qubits
        if self._conditions:
            i_cond_start = num_agg_qubits
            i_cond_end = i_cond_start + len(self._conditions) + 1 * (len(self._conditions) > 1)
            q_cond_target = [q_ancillas[i] for i in range(i_cond_start, i_cond_end)]

            # set remaining ancillas
            remaining_ancillas_start = i_cond_end
        else:
            # set remaining ancillas
            remaining_ancillas_start = num_agg_qubits

        q_rem_ancillas = [q_ancillas[i] for i in range(remaining_ancillas_start, len(q_ancillas))]

        # apply controlled or uncontrolled aggregation
        if not self._conditions:

            # apply aggregation
            self._aggregation_function.build(qc, q_agg, q_rem_ancillas)

            qc.barrier()

            # apply objective function
            self._univariate_objective.build(qc, q_obj, q_rem_ancillas)

            qc.barrier()

            # uncompute aggregation (all ancillas should be clean again now)
            self._aggregation_function.build_inverse(qc, q_agg, q_rem_ancillas)

            qc.barrier()

        else:

            if len(self._conditions) == 1:

                dimension = self._conditions[0][0]
                condition = self._conditions[0][1]

                i_condition_in_start = \
                    np.cumsum(self._uncertainty_model.num_qubits)[dimension] - \
                    self._uncertainty_model.num_qubits[dimension]
                i_condition_in_end = np.cumsum(self._uncertainty_model.num_qubits)[dimension]
                q_condition_in = \
                    [q_state[i] for i in range(i_condition_in_start, i_condition_in_end)]

                q_condition = q_condition_in + [q_cond_target[0]]

                condition.build(qc, q_condition, q_rem_ancillas)

                qc.barrier()

                # apply aggregation
                self._aggregation_function.build_controlled(qc,
                                                            q_agg,
                                                            q_cond_target[0],
                                                            q_rem_ancillas,
                                                            use_basis_gates=False)

                qc.barrier()

                # apply objective function
                self._univariate_objective.build(qc, q_obj, q_rem_ancillas)

                qc.barrier()

                # uncompute aggregation (all ancillas should be clean again now)
                self._aggregation_function.build_controlled_inverse(qc,
                                                                    q_agg,
                                                                    q_cond_target[0],
                                                                    q_rem_ancillas,
                                                                    use_basis_gates=False)

                qc.barrier()

                # uncompute condition
                condition.build_inverse(qc, q_condition, q_rem_ancillas)

            else:

                for j in range(len(self._conditions)):

                    dimension = self._conditions[j][0]
                    condition = self._conditions[j][1]

                    i_condition_in_start = \
                        np.cumsum(self._uncertainty_model.num_qubits)[dimension] - \
                        self._uncertainty_model.num_qubits[dimension]
                    i_condition_in_end = np.cumsum(self._uncertainty_model.num_qubits)[dimension]
                    q_condition_in = \
                        [q_state[i] for i in range(i_condition_in_start, i_condition_in_end)]

                    q_condition = q_condition_in + [q_cond_target[j]]

                    condition.build(qc, q_condition, q_rem_ancillas)

                qc.mct(q_cond_target[:-1], q_cond_target[-1], q_rem_ancillas)

                qc.barrier()

                # apply aggregation
                self._aggregation_function.build_controlled(qc,
                                                            q_agg,
                                                            q_cond_target[-1],
                                                            q_rem_ancillas, use_basis_gates=False)

                qc.barrier()

                # apply objective function
                self._univariate_objective.build(qc, q_obj, q_rem_ancillas)

                qc.barrier()

                # uncompute aggregation (all ancillas should be clean again now)
                self._aggregation_function.build_controlled_inverse(qc, q_agg, q_cond_target[-1],
                                                                    q_rem_ancillas,
                                                                    use_basis_gates=False)

                qc.barrier()

                qc.mct(q_cond_target[:-1], q_cond_target[-1], q_rem_ancillas)

                # uncompute condition
                for j in range(len(self._conditions)):

                    dimension = self._conditions[j][0]
                    condition = self._conditions[j][1]

                    i_condition_in_start = \
                        np.cumsum(self._uncertainty_model.num_qubits)[dimension] - \
                        self._uncertainty_model.num_qubits[dimension]
                    i_condition_in_end = np.cumsum(self._uncertainty_model.num_qubits)[dimension]
                    q_condition_in = \
                        [q_state[i] for i in range(i_condition_in_start, i_condition_in_end)]

                    q_condition = q_condition_in + [q_cond_target[j]]

                    condition.build_inverse(qc, q_condition, q_rem_ancillas)

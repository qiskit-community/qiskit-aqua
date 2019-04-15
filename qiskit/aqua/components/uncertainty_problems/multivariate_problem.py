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
import numpy as np


class MultivariateProblem(UncertaintyProblem):

    def __init__(self,
                 uncertainty_model,
                 aggregation_function,
                 univariate_objective,
                 conditions=None):
        """

        :param uncertainty_model:
        :param aggregation_function:
        :param univariate_objective:
        :param conditions:
        """

        # determine number of target qubits
        num_target_qubits = uncertainty_model.num_target_qubits + 1
        super().__init__(num_target_qubits)

        # set qubit indices for
        self.i_state = list(range(num_target_qubits-1))
        self.i_objective = num_target_qubits-1

        # store parameters
        self._uncertainty_model = uncertainty_model
        self._aggregation_function = aggregation_function
        self._univariate_objective = univariate_objective
        self._conditions = conditions

        self._params = {'i_state': range(num_target_qubits), 'i_objective': self.i_objective}

    def value_to_estimation(self, value):
        if hasattr(self._univariate_objective, 'value_to_estimation'):
            return self._univariate_objective.value_to_estimation(value)
        else:
            return value

    def required_ancillas(self):

        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_aggregation_ancillas = self._aggregation_function.num_sum_qubits + self._aggregation_function.required_ancillas()
        num_objective_ancillas = self._univariate_objective.required_ancillas()
        num_condition_ancillas = 0

        num_condition_target_ancillas = 0
        if self._conditions is not None and len(self._conditions) > 1:
            num_condition_target_ancillas += 1
        if self._conditions is not None:
            for (dim, condition) in self._conditions:
                num_condition_ancillas = np.maximum(num_condition_ancillas, num_condition_target_ancillas + condition.num_target_qubits + condition.required_ancillas())

        # aggregation, condition, and objective ancillas are added, since aggregation is only uncomputed AFTER objective evaluation
        return max([num_uncertainty_ancillas, num_aggregation_ancillas + num_objective_ancillas + num_condition_ancillas])

    def build(self, qc, q, q_ancillas=None, params=None):

        # apply uncertainty model (can use all ancillas and returns all clean)
        self._uncertainty_model.build(qc, q, q_ancillas, params)

        qc.barrier()

        # apply controlled or uncontrolled aggregation
        if self._conditions is None or len(self._conditions) == 0:

            # get all qubits up to the largest state qubit
            num_agg_qubits = self._aggregation_function.num_sum_qubits
            q_agg_in = q[self.i_state]
            q_agg_out = q_ancillas[[i for i in range(num_agg_qubits)]]
            q_agg = q_agg_in + q_agg_out

            # add required ancillas to auxilliary ancilla register
            i_agg_ancillas_start = self._aggregation_function.num_sum_qubits
            i_agg_ancillas_end = i_agg_ancillas_start + self._aggregation_function.required_ancillas()
            q_agg_ancillas = [q_ancillas[i] for i in range(i_agg_ancillas_start, i_agg_ancillas_end)]

            # determine objective qubits (aggregation qubits + objective qubit)
            q_obj = q_agg_out + [q[self.i_objective]]

            # determine remaining ancillas for objective (returns them clean)
            i_obj_ancillas_end = i_agg_ancillas_end + self._univariate_objective.required_ancillas()
            q_obj_ancillas = [q_ancillas[i] for i in range(i_agg_ancillas_end, i_obj_ancillas_end)]

            # apply aggregation
            self._aggregation_function.build(qc, q_agg, q_agg_ancillas)

            qc.barrier()

            # apply objective function
            self._univariate_objective.build(qc, q_obj, q_obj_ancillas, params)

            qc.barrier()

            # uncompute aggregation (all ancillas should be clean again now)
            self._aggregation_function.build_inverse(qc, q_agg, q_agg_ancillas)

            qc.barrier()

        else:

            if len(self._conditions) == 1:

                # get all qubits up to the largest state qubit
                num_agg_qubits = self._aggregation_function.num_sum_qubits
                q_agg_in = q[self.i_state]
                q_agg_out = q_ancillas[[i for i in range(num_agg_qubits)]]
                q_agg = q_agg_in + q_agg_out

                # add required ancillas to auxilliary ancilla register
                i_agg_ancillas_start = self._aggregation_function.num_sum_qubits
                i_agg_ancillas_end = i_agg_ancillas_start + self._aggregation_function.required_ancillas_controlled()
                q_agg_ancillas = [q_ancillas[i] for i in range(i_agg_ancillas_start, i_agg_ancillas_end)]

                # determine objective qubits (aggregation qubits + objective qubit)
                q_obj = q_agg_out + [q[self.i_objective]]

                # determine remaining ancillas for objective (returns them clean)
                i_obj_ancillas_end = i_agg_ancillas_end + self._univariate_objective.required_ancillas()
                q_obj_ancillas = [q_ancillas[i] for i in range(i_agg_ancillas_end, i_obj_ancillas_end)]

                dimension = self._conditions[0][0]
                condition = self._conditions[0][1]

                i_condition_in_end = i_obj_ancillas_end+self._uncertainty_model.num_qubits[dimension]
                q_condition_in = q_ancillas[i_obj_ancillas_end:i_condition_in_end]
                q_condition_out = q_ancillas[i_condition_in_end]
                q_condition = q_condition_in + [q_condition_out]

                i_cond_ancillas_start = i_condition_in_end + 1
                i_cond_ancillas_end = i_cond_ancillas_start + condition.required_ancillas()
                q_condition_ancillas = q_ancillas[i_cond_ancillas_start:i_cond_ancillas_end]

                condition.build(qc, q_condition, q_condition_ancillas)

                qc.barrier()

                # apply aggregation
                self._aggregation_function.build_controlled(qc, q_agg, q_condition_out, q_agg_ancillas)

                qc.barrier()

                # apply objective function
                self._univariate_objective.build(qc, q_obj, q_obj_ancillas)

                qc.barrier()

                # uncompute aggregation (all ancillas should be clean again now)
                self._aggregation_function.build_controlled_inverse(qc, q_agg, q_condition_out, q_agg_ancillas)

                qc.barrier()

                # uncompute condition
                condition.build_inverse(qc, q_condition, q_condition_ancillas)

            else:

                # combine conditions in ancilla
                qc.mct([q[i] for i in i_condition], q_ancillas[-1], None)

                # aggregate results controlled with ancilla
                self._aggregation_function.build_controlled(qc, q, q_ancillas[-1], q_ancillas)

                # uncompute conditions in ancilla
                qc.mct([q[i] for i in i_condition], q_ancillas[-1], None)




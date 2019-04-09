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
The European Call Option Expected Value.
"""
import numpy as np
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.circuits.gates import *
from qiskit.aqua.components.uncertainty_problems.fixed_value_comparator import FixedValueComparator


class EuropeanCallExpectedValue(UncertaintyProblem):
    """
    The European Call Option Expected Value.

    Evaluates the expected payoff for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    CONFIGURATION = {
        'name': 'EuropeanCallExpectedValue',
        'description': 'European Call Expected Value',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ECEV_schema',
            'type': 'object',
            'properties': {
                'strike_price': {
                    'type': 'integer',
                    'default': 0
                },
                'c_approx': {
                    'type': 'number',
                    'default': 0.5
                },
                'i_state': {
                    'type': ['array', 'null'],
                    'items': {
                        'type': 'integer'
                    },
                    'default': None
                },
                'i_compare': {
                    'type': ['integer', 'null'],
                    'default': None
                },
                'i_objective': {
                    'type': ['integer', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'univariate_distribution',
                'default': {
                    'name': 'NormalDistribution'
                }
            },
        ],
    }

    def __init__(self, uncertainty_model, strike_price, c_approx, i_state=None, i_compare=None, i_objective=None):
        super().__init__(uncertainty_model.num_target_qubits + 2)

        self._uncertainty_model = uncertainty_model
        self._strike_price = strike_price
        self._c_approx = c_approx

        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        if i_compare is None:
            i_compare = uncertainty_model.num_target_qubits
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits + 1

        self._params = {
            'i_state': i_state,
            'i_compare': i_compare,
            'i_objective': i_objective
        }

        super().validate(locals())

        # map strike price to {0, ..., 2^n-1}
        lb = uncertainty_model.low
        ub = uncertainty_model.high
        self._mapped_strike_price = int(np.round((strike_price - lb)/(ub - lb) * (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = FixedValueComparator(uncertainty_model.num_target_qubits + 1, self._mapped_strike_price)

        self.offset_angle_zero = np.pi / 4 * (1 - self._c_approx)
        if self._mapped_strike_price < uncertainty_model.num_values - 1:
            self.offset_angle = -1 * np.pi / 2 * self._c_approx * self._mapped_strike_price / (uncertainty_model.num_values - self._mapped_strike_price - 1)
            self.slope_angle = np.pi / 2 * self._c_approx / (uncertainty_model.num_values - self._mapped_strike_price - 1)
        else:
            self.offset_angle = 0
            self.slope_angle = 0

    def value_to_estimation(self, value):
        estimator = value - 1 / 2 + np.pi / 4 * self._c_approx
        estimator *= 2 / np.pi / self._c_approx
        estimator *= (self._uncertainty_model.num_values - self._mapped_strike_price - 1)
        estimator *= (self._uncertainty_model.high - self._uncertainty_model.low) / (self._uncertainty_model.num_values - 1)
        return estimator

    def required_ancillas(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_comparator_ancillas = self._comparator.required_ancillas()
        num_ancillas = num_uncertainty_ancillas + num_comparator_ancillas
        return num_ancillas

    def required_ancillas_controlled(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas_controlled()
        num_comparator_ancillas = self._comparator.required_ancillas_controlled()
        num_ancillas_controlled = num_uncertainty_ancillas + num_comparator_ancillas
        return num_ancillas_controlled

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # get qubits
        q_compare = q[params['i_compare']]
        q_objective = q[params['i_objective']]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q, q_ancillas, params)

        # apply comparator to compare qubit
        self._comparator.build(qc, q, q_ancillas, params)

        # apply approximate payoff function
        qc.ry(2 * self.offset_angle_zero, q_objective)
        qc.cry(2 * self.offset_angle, q_compare, q_objective)
        for i in self._params['i_state']:
            qc.mcry(2 * self.slope_angle * 2 ** i, [q_compare, q[i]], q_objective, None)

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # get qubits
        q_compare = q[params['i_compare']]
        q_objective = q[params['i_objective']]

        # apply approximate payoff function
        qc.ry(-2 * self.offset_angle_zero, q_objective)
        qc.cry(-2 * self.offset_angle, q_compare, q_objective)
        for i in self._params['i_state']:
            qc.mcry(-2 * self.slope_angle * 2 ** i, [q_compare, q[i]], q_objective, None)

        # apply comparator to compare qubit
        self._comparator.build_inverse(qc, q, q_ancillas, params)

        # apply uncertainty model
        self._uncertainty_model.build_inverse(qc, q, q_ancillas, params)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # get qubits
        q_compare = q[params['i_compare']]
        q_objective = q[params['i_objective']]

        # apply uncertainty model
        self._uncertainty_model.build_controlled(qc, q, q_control, q_ancillas, params)

        # apply comparator to compare qubit
        self._comparator.build_controlled(qc, q, q_control, q_ancillas, params)

        # apply approximate payoff function
        qc.cry(2 * self.offset_angle_zero, q_control, q_objective)
        qc.mcry(2 * self.offset_angle, [q_control, q_compare], q_objective, None)
        for i in self._params['i_state']:
            qc.mcry(2 * self.slope_angle * 2 ** i, [q_control, q_compare, q[i]], q_objective, q_ancillas)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # get qubits
        q_compare = q[params['i_compare']]
        q_objective = q[params['i_objective']]

        # apply approximate payoff function
        qc.cry(-2 * self.offset_angle_zero, q_control, q_objective)
        mcry(-2 * self.offset_angle, [q_control, q_compare], q_objective, None, qc)
        for i in self._params['i_state']:
            mcry(-2 * self.slope_angle * 2 ** i, [q_control, q_compare, q[i]], q_objective, q_ancillas, qc)

        # apply comparator to compare qubit
        self._comparator.build_controlled_inverse(qc, q, q_control, q_ancillas, params)

        # apply uncertainty model
        self._uncertainty_model.build_controlled_inverse(qc, q, q_control, q_ancillas, params)

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
The European Call Option Delta.
"""

import numpy as np
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.components.uncertainty_problems.fixed_value_comparator import FixedValueComparator


class EuropeanCallDelta(UncertaintyProblem):
    """
    The European Call Option Delta.

    Evaluates the variance for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    CONFIGURATION = {
        'name': 'EuropeanCallDelta',
        'description': 'European Call Delta',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ECD_schema',
            'type': 'object',
            'properties': {
                'strike_price': {
                    'type': 'integer',
                    'default': 0
                },
                'i_state': {
                    'type': ['array', 'null'],
                    'items': {
                        'type': 'integer'
                    },
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

    def __init__(self, uncertainty_model, strike_price, i_state=None, i_objective=None):
        super().__init__(uncertainty_model.num_target_qubits + 1)

        self._uncertainty_model = uncertainty_model
        self._strike_price = strike_price

        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits

        self._params = {
            'i_state': i_state,
            'i_compare': i_objective,  # compare and objective qubits are the same for the delta.
            'i_objective': i_objective
        }

        super().validate(locals())

        # map strike price to {0, ..., 2^n-1}
        lb = uncertainty_model.low
        ub = uncertainty_model.high
        self._mapped_strike_price = int(np.ceil((strike_price - lb)/(ub - lb) * (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = FixedValueComparator(uncertainty_model.num_target_qubits + 1, self._mapped_strike_price)

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

        # apply uncertainty model
        self._uncertainty_model.build(qc, q, q_ancillas, params)

        # apply comparator to compare qubit
        self._comparator.build(qc, q, q_ancillas, params)

    def build_inverse(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # apply comparator to compare qubit
        self._comparator.build_inverse(qc, q, q_ancillas, params)

        # apply uncertainty model
        self._uncertainty_model.build_inverse(qc, q, q_ancillas, params)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # apply uncertainty model
        self._uncertainty_model.build_controlled(qc, q, q_control, q_ancillas, params)

        # apply comparator to compare qubit
        self._comparator.build_controlled(qc, q, q_control, q_ancillas, params)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # apply comparator to compare qubit
        self._comparator.build_controlled_inverse(qc, q, q_control, q_ancillas, params)

        # apply uncertainty model
        self._uncertainty_model.build_controlled_inverse(qc, q, q_control, q_ancillas, params)

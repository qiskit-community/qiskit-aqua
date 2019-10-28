# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The European Call Option Delta.
"""

import numpy as np
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.circuits.fixed_value_comparator import FixedValueComparator

# pylint: disable=invalid-name


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
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'ECD_schema',
            'type': 'object',
            'properties': {
                'strike_price': {
                    'type': 'number',
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
        """
        Constructor.

        Args:
            uncertainty_model (UnivariateDistribution): uncertainty model for spot price
            strike_price (float): strike price of the European option
            i_state (Union(list, numpy.ndarray)): indices of qubits representing the uncertainty
            i_objective (int): index of qubit for objective function
        """
        super().__init__(uncertainty_model.num_target_qubits + 1)

        self._uncertainty_model = uncertainty_model
        self._strike_price = strike_price

        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        self.i_state = i_state
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits
        self.i_objective = i_objective

        super().validate(locals())

        # map strike price to {0, ..., 2^n-1}
        lb = uncertainty_model.low
        ub = uncertainty_model.high
        self._mapped_strike_price = int(np.ceil((strike_price - lb) /
                                                (ub - lb) * (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = FixedValueComparator(uncertainty_model.num_target_qubits,
                                                self._mapped_strike_price)

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

        # get qubit lists
        q_state = [q[i] for i in self.i_state]
        q_objective = q[self.i_objective]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        # apply comparator to compare qubit
        self._comparator.build(qc, q_state + [q_objective], q_ancillas)

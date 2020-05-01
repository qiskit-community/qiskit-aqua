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
The European Call Option Expected Value.
"""

import numpy as np
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.circuits.fixed_value_comparator import FixedValueComparator

# pylint: disable=invalid-name


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
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'ECEV_schema',
            'type': 'object',
            'properties': {
                'strike_price': {
                    'type': 'number',
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

    def __init__(self, uncertainty_model, strike_price, c_approx, i_state=None,
                 i_compare=None, i_objective=None):
        """
        Constructor.

        Args:
            uncertainty_model (UnivariateDistribution): uncertainty model for spot price
            strike_price (float): strike price of the European option
            c_approx (float): approximation factor for linear payoff
            i_state (Optional(Union(list, numpy.ndarray))): indices of qubits
                                                            representing the uncertainty
            i_compare (Optional(int)): index of qubit for comparing spot price to strike price
                            (enabling payoff or not)
            i_objective (Optional(int)): index of qubit for objective function
        """
        super().__init__(uncertainty_model.num_target_qubits + 2)

        self._uncertainty_model = uncertainty_model
        self._strike_price = strike_price
        self._c_approx = c_approx

        if i_state is None:
            i_state = list(range(uncertainty_model.num_target_qubits))
        self.i_state = i_state
        if i_compare is None:
            i_compare = uncertainty_model.num_target_qubits
        self.i_compare = i_compare
        if i_objective is None:
            i_objective = uncertainty_model.num_target_qubits + 1
        self.i_objective = i_objective

        super().validate(locals())

        # map strike price to {0, ..., 2^n-1}
        lb = uncertainty_model.low
        ub = uncertainty_model.high
        self._mapped_strike_price = int(np.round((strike_price - lb) /
                                                 (ub - lb) * (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = FixedValueComparator(uncertainty_model.num_target_qubits,
                                                self._mapped_strike_price)

        self.offset_angle_zero = np.pi / 4 * (1 - self._c_approx)
        if self._mapped_strike_price < uncertainty_model.num_values - 1:
            self.offset_angle = -1 * np.pi / 2 * self._c_approx * self._mapped_strike_price / \
                        (uncertainty_model.num_values - self._mapped_strike_price - 1)
            self.slope_angle = np.pi / 2 * self._c_approx / \
                (uncertainty_model.num_values - self._mapped_strike_price - 1)
        else:
            self.offset_angle = 0
            self.slope_angle = 0

    def value_to_estimation(self, value):
        estimator = value - 1 / 2 + np.pi / 4 * self._c_approx
        estimator *= 2 / np.pi / self._c_approx
        estimator *= (self._uncertainty_model.num_values - self._mapped_strike_price - 1)
        estimator *= (self._uncertainty_model.high - self._uncertainty_model.low) / \
            (self._uncertainty_model.num_values - 1)
        return estimator

    def required_ancillas(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_comparator_ancillas = self._comparator.required_ancillas()
        num_ancillas = int(np.maximum(num_uncertainty_ancillas, num_comparator_ancillas))
        return num_ancillas

    def build(self, qc, q, q_ancillas=None, params=None):

        # get qubits
        q_state = [q[i] for i in self.i_state]
        q_compare = q[self.i_compare]
        q_objective = q[self.i_objective]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        # apply comparator to compare qubit
        self._comparator.build(qc, q_state + [q_compare], q_ancillas)

        # apply approximate payoff function
        qc.ry(2 * self.offset_angle_zero, q_objective)
        qc.cry(2 * self.offset_angle, q_compare, q_objective)
        for i, qi in enumerate(q_state):
            qc.mcry(2 * self.slope_angle * 2 ** i, [q_compare, qi], q_objective, None)

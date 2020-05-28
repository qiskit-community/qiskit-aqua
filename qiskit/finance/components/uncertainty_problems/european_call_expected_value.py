# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The European Call Option Expected Value."""

from typing import Optional, Union, List
import numpy as np
from qiskit.circuit.library import IntegerComparator
from qiskit.aqua.components.uncertainty_models import UnivariateDistribution
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem


class EuropeanCallExpectedValue(UncertaintyProblem):
    """The European Call Option Expected Value.

    Evaluates the expected payoff for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(self,
                 uncertainty_model: UnivariateDistribution,
                 strike_price: float,
                 c_approx: float,
                 i_state: Optional[Union[List[int], np.ndarray]] = None,
                 i_compare: Optional[int] = None,
                 i_objective: Optional[int] = None) -> None:
        """
        Constructor.

        Args:
            uncertainty_model: uncertainty model for spot price
            strike_price: strike price of the European option
            c_approx: approximation factor for linear payoff
            i_state: indices of qubits representing the uncertainty
            i_compare: index of qubit for comparing spot price to strike price
                            (enabling payoff or not)
            i_objective: index of qubit for objective function
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

        # map strike price to {0, ..., 2^n-1}
        lower = uncertainty_model.low
        upper = uncertainty_model.high
        self._mapped_strike_price = int(np.round((strike_price - lower) /
                                                 (upper - lower) *
                                                 (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = IntegerComparator(uncertainty_model.num_target_qubits,
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
        num_comparator_ancillas = self._comparator.num_ancilla_qubits
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
        qubits = q_state[:] + [q_compare]
        if q_ancillas:
            qubits += q_ancillas[:self._comparator.num_ancilla_qubits]
        qc.append(self._comparator.to_instruction(), qubits)

        # apply approximate payoff function
        qc.ry(2 * self.offset_angle_zero, q_objective)
        qc.cry(2 * self.offset_angle, q_compare, q_objective)
        for i, q_i in enumerate(q_state):
            qc.mcry(2 * self.slope_angle * 2 ** i, [q_compare, q_i], q_objective, None)

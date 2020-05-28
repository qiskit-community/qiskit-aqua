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

"""The European Call Option Delta."""

from typing import Optional, Union, List
import numpy as np
from qiskit.circuit.library import IntegerComparator
from qiskit.aqua.components.uncertainty_models import UnivariateDistribution
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem

# pylint: disable=invalid-name


class EuropeanCallDelta(UncertaintyProblem):
    """The European Call Option Delta.

    Evaluates the variance for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(self,
                 uncertainty_model: UnivariateDistribution,
                 strike_price: float,
                 i_state: Optional[Union[List[int], np.ndarray]] = None,
                 i_objective: Optional[int] = None) -> None:
        """
        Constructor.

        Args:
            uncertainty_model: uncertainty model for spot price
            strike_price: strike price of the European option
            i_state: indices of qubits representing the uncertainty
            i_objective: index of qubit for objective function
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

        # map strike price to {0, ..., 2^n-1}
        lb = uncertainty_model.low
        ub = uncertainty_model.high
        self._mapped_strike_price = \
            int(np.ceil((strike_price - lb) / (ub - lb) * (uncertainty_model.num_values - 1)))

        # create comparator
        self._comparator = IntegerComparator(uncertainty_model.num_target_qubits,
                                             self._mapped_strike_price)

    def required_ancillas(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas()
        num_comparator_ancillas = self._comparator.num_ancilla_qubits
        num_ancillas = num_uncertainty_ancillas + num_comparator_ancillas
        return num_ancillas

    def required_ancillas_controlled(self):
        num_uncertainty_ancillas = self._uncertainty_model.required_ancillas_controlled()
        num_comparator_ancillas = self._comparator.num_ancilla_qubits
        num_ancillas_controlled = num_uncertainty_ancillas + num_comparator_ancillas
        return num_ancillas_controlled

    def build(self, qc, q, q_ancillas=None, params=None):
        # get qubit lists
        q_state = [q[i] for i in self.i_state]
        q_objective = q[self.i_objective]

        # apply uncertainty model
        self._uncertainty_model.build(qc, q_state, q_ancillas)

        # apply comparator to compare qubit
        qubits = q_state[:] + [q_objective]
        if q_ancillas:
            qubits += q_ancillas[:self.required_ancillas()]
        qc.append(self._comparator.to_instruction(), qubits)

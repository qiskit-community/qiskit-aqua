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

from typing import Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator


class EuropeanCallExpectedValue(QuantumCircuit):
    """The European Call Option Expected Value.

    Evaluates the expected payoff for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(self,
                 num_state_qubits: int,
                 strike_price: float,
                 rescaling_factor: float,
                 bounds: Tuple[float, float],
                 ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits used to represent the random variable.
            strike_price: strike price of the European option
            rescaling_factor: approximation factor for linear payoff
            bounds: The bounds of the discretized random variable.
        """
        # create comparator
        comparator = IntegerComparator(num_state_qubits, strike_price)

        qr_state = QuantumRegister(num_state_qubits, 'state')
        qr_compare = QuantumRegister(1, 'compare')
        qr_objective = QuantumRegister(1, 'objective')
        super().__init__(qr_state, qr_compare, qr_objective)
        if comparator.num_ancillas > 0:
            qr_work = QuantumRegister(comparator.num_ancillas, 'work')
            self.add_register(qr_work)
        else:
            qr_work = []

        # map strike price to {0, ..., 2^n-1}
        num_values = 2 ** num_state_qubits
        strike_price = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (num_values - 1)
        strike_price = int(np.round(strike_price))

        offset_angle_zero = np.pi / 4 * (1 - rescaling_factor)
        if strike_price < num_values - 1:
            offset_angle = -np.pi / 2 * rescaling_factor * strike_price / \
                (num_values - strike_price - 1)
            slope_angle = np.pi / 2 * rescaling_factor / (num_values - strike_price - 1)
        else:
            offset_angle = 0
            slope_angle = 0

        self._rescaling_factor = rescaling_factor
        self._num_values = num_values
        self._bounds = bounds
        self._strike_price = strike_price

        # apply comparator to compare qubit
        self.append(comparator.to_instruction(), qr_state[:] + qr_compare[:] + qr_work[:])

        # apply approximate payoff function
        self.ry(2 * offset_angle_zero, qr_objective)
        self.cry(2 * offset_angle, qr_compare, qr_objective)
        for i, q_i in enumerate(qr_state):
            # pylint: disable=no-member
            self.mcry(2 * slope_angle * 2 ** i, qr_compare[:] + [q_i], qr_objective, None)

        # TODO is this the same as
        # self.append(comparator)
        # self.ry(2 * offset_angle_zero)  # why is this not controlled??
        # f = LinearAmplitudeFunction(num_state_qubits, slope=1, offset=(c - 1)/2,
        #                             domain=bounds, image=(s, 2^n - 1))
        # self.append(f.control())

    def post_processing(self, scaled_value: float) -> float:
        """Map the scaled value back to the original domain.

        Args:
            scaled_value: The scaled value.

        Returns:
            The scaled value mapped back to the original domain.
        """
        value = scaled_value - 1 / 2 + np.pi / 4 * self._rescaling_factor
        value *= 2 / np.pi / self._rescaling_factor
        value *= (self._num_values - self._strike_price - 1)
        value *= (self._bounds[1] - self._bounds[0]) / (self._num_values - 1)
        return value

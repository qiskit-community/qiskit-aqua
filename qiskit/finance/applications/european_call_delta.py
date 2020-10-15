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

from typing import Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import IntegerComparator


class EuropeanCallDelta(QuantumCircuit):
    """The European Call Option Delta.

    Evaluates the variance for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(self, num_state_qubits: int, strike_price: float, bounds: Tuple[float, float]
                 ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits used to encode the random variable.
            strike_price: strike price of the European option
            bounds: The bounds of the discretized random variable.
        """
        # map strike price to {0, ..., 2^n-1}
        num_values = 2 ** num_state_qubits
        strike_price = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (num_values - 1)
        strike_price = int(np.ceil(strike_price))

        # create comparator
        comparator = IntegerComparator(num_state_qubits, strike_price)

        # initialize circuit
        qr_state = QuantumRegister(comparator.num_qubits - comparator.num_ancillas, 'state')
        qr_work = QuantumRegister(comparator.num_ancillas, 'work')
        super().__init__(qr_state, qr_work, name='ECD')

        self.append(comparator.to_gate(), self.qubits)

    def post_processing(self, scaled_value: float) -> float:
        """Map the scaled value back to the original domain.

        Args:
            scaled_value: The scaled value.

        Returns:
            The scaled value mapped back to the original domain.
        """
        return scaled_value

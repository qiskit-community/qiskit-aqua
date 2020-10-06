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

"""The Fixed Income Expected Value function."""

from typing import List, Tuple
import numpy as np
from qiskit import QuantumCircuit


class FixedIncomeExpectedValue(QuantumCircuit):
    """The Fixed Income Expected Value.

    Evaluates a fixed income asset with uncertain interest rates.
    """

    def __init__(self,
                 num_qubits: List[int],
                 pca_matrix: np.ndarray,
                 initial_interests: List[int],
                 cash_flow: List[float],
                 rescaling_factor: float,
                 bounds: List[Tuple[float, float]],
                 ) -> None:
        """
        Args:
            num_qubits: TODO
            pca_matrix: PCA matrix for delta_r (changes in interest rates)
            initial_interests: offset for interest rates (= initial interest rates)
            cash_flow: cash flow time series
            rescaling_factor: approximation scaling factor
            bounds: TODO
        """
        self._rescaling_factor = rescaling_factor

        if not isinstance(pca_matrix, np.ndarray):
            pca_matrix = np.asarray(pca_matrix)

        # get number of time steps
        time_steps = len(cash_flow)

        # get dimension of uncertain model
        dimensions = len(num_qubits)

        # initialize parent class
        super().__init__(sum(num_qubits) + 1)

        # construct PCA-based cost function (1st order approximation):
        # c_t / (1 + A_t x + b_t)^{t+1} ~ c_t / (1 + b_t)^{t+1} - (t+1) c_t A_t /
        # (1 + b_t)^{t+2} x = h + np.dot(g, x)
        # pylint: disable=invalid-name
        h = 0
        g = np.zeros(dimensions)
        for t in range(time_steps):
            h += cash_flow[t] / (1 + initial_interests[t]) ** (t + 1)
            g += -(t + 1) * cash_flow[t] * pca_matrix[t, :] / (1 + initial_interests[t]) ** (t + 2)

        # compute overall offset using lower bound for x (corresponding to x = min)
        low = [bound[0] for bound in bounds]
        offset = np.dot(low, g) + h

        # compute overall slope
        slopes = []
        for k in range(dimensions):
            n_k = num_qubits[k]
            for i in range(n_k):
                slope = 2 ** i / (2 ** n_k - 1) * (bounds[k][1] - bounds[k][0]) * g[k]
                slopes += [slope]

        # evaluate min and max values
        # for scaling to [0, 1] is then given by (V - min) / (max - min)
        min_value = offset + sum(slopes)
        max_value = offset

        # store image for post_processing
        self._image = [min_value, max_value]

        # reset offset / slope accordingly
        offset -= min_value
        offset /= max_value - min_value
        slopes /= max_value - min_value

        # apply approximation scaling
        offset_angle = (offset - 1 / 2) * np.pi / 2 * rescaling_factor + np.pi / 4
        slope_angles = slopes * np.pi / 2 * rescaling_factor

        # apply approximate payoff function
        self.ry(2 * offset_angle, self.num_qubits - 1)
        for i, angle in enumerate(slope_angles):
            self.cry(2 * angle, i, self.num_qubits - 1)

    def post_processing(self, scaled_value: float) -> float:
        """Map the scaled value back to the original domain.

        Args:
            scaled_value: The scaled value.

        Returns:
            The scaled value mapped back to the original domain.
        """
        value = scaled_value - 1 / 2
        value *= 2 / np.pi / self._rescaling_factor
        value += 1 / 2
        value *= self._image[1] - self._image[0]
        value += self._image[0]
        return value

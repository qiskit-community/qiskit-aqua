# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
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
    r"""The Fixed Income Expected Value amplitude function.

    This circuit can be used to evaluate the expected value of the total value :math:`V` of the
    assets

    .. math::

        V = \sum_{t=1}^T \frac{c_t}{(1+r_t)^t}.

    Here :math:`c_t` are the cash flows of the assets and :math:`r_t` are the interest rates.
    The interest rates are subject to uncertainty and can be described by a PCA-decomposition
    into the ``pca_matrix`` :math:`A` and ``initial_interests` :math:`\vec{b}`. For a sample
    :math:`\vec{x}` of a random variable, the interest rates are modeled as:

    .. math::

        \vec{r} = A \vec{x} + \vec{b}.

    The number of qubits used to represent each asset is specified by ``num_qubits`` and the
    bounds of the random variable by ``bounds``.

    The approximation of the objective function follows [1].

    References:

        [1]: Woerner, S., & Egger, D. J. (2018).
             Quantum Risk Analysis.
             `arXiv:1806.06893 <http://arxiv.org/abs/1806.06893>`_

    """

    def __init__(self,
                 num_qubits: List[int],
                 pca_matrix: np.ndarray,
                 initial_interests: List[int],
                 cash_flow: List[float],
                 rescaling_factor: float,
                 bounds: List[Tuple[float, float]],
                 ) -> None:
        r"""
        Args:
            num_qubits: A list specifying the number of qubits used to discretize the assets.
            pca_matrix: The PCA matrix for the changes in the interest rates, :math:`\delta_r`.
            initial_interests: The initial interest rates / offsets for the interest rates.
            cash_flow: The cash flow time series.
            rescaling_factor: The scaling factor used in the Taylor approximation.
            bounds: The bounds for return values the assets can attain.
        """
        self._rescaling_factor = rescaling_factor

        if not isinstance(pca_matrix, np.ndarray):
            pca_matrix = np.asarray(pca_matrix)

        # get number of time steps
        time_steps = len(cash_flow)

        # get the number of assets
        dimensions = len(num_qubits)

        # initialize parent class
        super().__init__(sum(num_qubits) + 1, name='F')

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
        slope_angles = slopes * np.pi / 2 * rescaling_factor  # type: ignore

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

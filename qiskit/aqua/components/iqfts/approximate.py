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

"""The Approximate IQFT."""

from qiskit.aqua.circuits import FourierTransformCircuits as ftc
from qiskit.aqua.utils.validation import validate_min
from . import IQFT


class Approximate(IQFT):
    """
    The Approximate IQFT.

    This form of IQFT generates the inverse of an Approximate Quantum Fourier Transform as
    described in https://arxiv.org/abs/1803.04933.
    """

    def __init__(self,
                 num_qubits: int,
                 degree: int = 0) -> None:
        """
        Args:
            num_qubits: The number of qubits
            degree: The degree of approximation. 0 is the minimum value and causes no
                approximation so will in fact be the same as a
                :class:`~qiskit.aqua.components.iqfts.StandardIQFT`.
        """
        validate_min('num_qubits', num_qubits, 1)
        validate_min('degree', degree, 0)
        super().__init__()
        self._num_qubits = num_qubits
        self._degree = degree

    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        return ftc.construct_circuit(
            circuit=circuit,
            qubits=qubits,
            inverse=True,
            approximation_degree=self._degree,
            do_swaps=do_swaps
        )

    def _build_matrix(self):
        raise NotImplementedError

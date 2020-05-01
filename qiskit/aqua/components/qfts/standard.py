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

"""The Standard QFT."""

from scipy import linalg

from qiskit.aqua.utils.validation import validate_min
from .approximate import Approximate


class Standard(Approximate):
    """The Standard QFT.

    This is a standard Quantum Fourier Transform
    """

    def __init__(self, num_qubits: int) -> None:
        """
        Args:
            num_qubits: The number of qubits
        """
        validate_min('num_qubits', num_qubits, 1)
        super().__init__(num_qubits, degree=0)

    def _build_matrix(self):
        # pylint: disable=no-member
        return linalg.inv(linalg.dft(2 ** self._num_qubits, scale='sqrtn'))

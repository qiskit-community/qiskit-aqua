# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Watson Hamiltonian """

from typing import Union, List


class WatsonHamiltonian:
    """
    Watson Hamiltonian class containing the results of a driver's anharmonic calculation
    """

    def __init__(self, data: List[List[Union[int, float]]], num_modes: int):
        """
        Args:
            data: Hamiltonian matrix elements
            num_modes: number of modes
        """
        self._data = data
        self._num_modes = num_modes

    @property
    def data(self) -> List[List[Union[int, float]]]:
        """Returns the matrix elements of the Hamiltonian"""
        return self._data

    @property
    def num_modes(self) -> int:
        """Returns the number of modes"""
        return self._num_modes

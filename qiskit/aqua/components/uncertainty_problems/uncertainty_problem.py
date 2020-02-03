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

"""
The abstract Uncertainty Problem component.
"""

from abc import ABC
from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.utils.validation import validate_min

# pylint: disable=abstract-method


class UncertaintyProblem(CircuitFactory, ABC):
    """
    The abstract Uncertainty Problem component.
    """

    # pylint: disable=useless-super-delegation
    def __init__(self, num_qubits: int) -> None:
        validate_min('num_qubits', num_qubits, 1)
        super().__init__(num_qubits)

    def value_to_estimation(self, value):
        """ value to estimate """
        return value

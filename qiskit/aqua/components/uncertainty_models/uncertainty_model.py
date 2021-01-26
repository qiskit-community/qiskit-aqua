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
This module contains the definition of a base class for
uncertainty models. An uncertainty model could be used for
constructing Amplification Estimation tasks.
"""

from abc import ABC, abstractmethod
from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.utils import CircuitFactory


class UncertaintyModel(CircuitFactory, ABC):
    """
    The abstract Uncertainty Model
    """

    __REPLACEMENT = 'a qiskit.QuantumCircuit'

    # pylint: disable=useless-super-delegation
    def __init__(self, num_target_qubits: int) -> None:
        validate_min('num_target_qubits', num_target_qubits, 1)
        super().__init__(num_target_qubits)

    @abstractmethod
    def build(self, qc, q, q_ancillas=None, params=None):
        raise NotImplementedError()

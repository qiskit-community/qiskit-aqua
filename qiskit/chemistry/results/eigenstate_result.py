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

"""Eigenstate results module."""

from typing import Optional, List, Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.aqua.algorithms import AlgorithmResult
from qiskit.aqua.operators import OperatorBase


class EigenstateResult(AlgorithmResult):
    """The eigenstate result interface."""

    @property
    def eigenenergies(self) -> Optional[np.ndarray]:
        """ returns eigen energies """
        return self.get('eigenenergies')

    @eigenenergies.setter
    def eigenenergies(self, value: np.ndarray) -> None:
        """ set eigen energies """
        self.data['eigenenergies'] = value

    @property
    def eigenstates(self) -> Optional[List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                                 QuantumCircuit, Instruction, OperatorBase]]]:
        """ returns eigen states """
        return self.get('eigenstates')

    @eigenstates.setter
    def eigenstates(self, value: List[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]) -> None:
        """ set eigen states """
        self.data['eigenstates'] = value

    @property
    def groundenergy(self) -> Optional[float]:
        """ returns ground energy """
        energies = self.get('eigenenergies')
        if energies:
            return energies[0].real
        return None

    @property
    def groundstate(self) -> Optional[Union[str, dict, Result, list, np.ndarray, Statevector,
                                            QuantumCircuit, Instruction, OperatorBase]]:
        """ returns ground state """
        states = self.get('eigenstates')
        if states:
            return states[0]
        return None

    @property
    def aux_operator_eigenvalues(self) -> Optional[List[float]]:
        """ return aux operator eigen values """
        return self.get('aux_operator_eigenvalues')

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: List[float]) -> None:
        """ set aux operator eigen values """
        self.data['aux_operator_eigenvalues'] = value

    @property
    def raw_result(self) -> Optional[AlgorithmResult]:
        """Returns the raw algorithm result."""
        return self.get('raw_result')

    @raw_result.setter
    def raw_result(self, result: AlgorithmResult) -> None:
        self.data['raw_result'] = result

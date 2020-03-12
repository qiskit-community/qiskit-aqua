# -*- coding: utf-8 -*-

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

"""GroverOptimizationResults module"""

from typing import Dict, Tuple, Union
from qiskit.optimization.results import OptimizationResult


class GroverOptimizationResults(OptimizationResult):

    """A results object for Grover Optimization methods."""

    def __init__(self, x, fval, operation_counts: Dict[int, Dict[str, int]], rotations: int,
                 n_input_qubits: int, n_output_qubits: int,
                 func_dict: Dict[Union[int, Tuple[int, int]], int]) -> None:
        """
        Args:
            operation_counts: The counts of each operation performed per iteration.
            rotations: The total number of Grover rotations performed.
            n_input_qubits: The number of qubits used to represent the input.
            n_output_qubits: The number of qubits used to represent the output.
            func_dict: A dictionary representation of the function, where the keys correspond
                to a variable, and the values are the corresponding coefficients.
        """
        super().__init__(x, fval)
        self._operation_counts = operation_counts
        self._rotations = rotations
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._func_dict = func_dict

    @property
    def operation_counts(self) -> Dict[int, Dict[str, int]]:
        """Getter of operation_counts"""
        return self._operation_counts

    @property
    def rotation_count(self) -> int:
        """Getter of rotation_count"""
        return self._rotations

    @property
    def n_input_qubits(self) -> int:
        """Getter of n_input_qubits"""
        return self._n_input_qubits

    @property
    def n_output_qubits(self) -> int:
        """Getter of n_output_qubits"""
        return self._n_output_qubits

    @property
    def func_dict(self) -> Dict[Union[int, Tuple[int, int]], int]:
        """Getter of func_dict"""
        return self._func_dict

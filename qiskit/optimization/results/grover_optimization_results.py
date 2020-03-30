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


class GroverOptimizationResults:
    """A results object for Grover Optimization methods."""

    def __init__(self, operation_counts: Dict[int, Dict[str, int]],
                 n_input_qubits: int, n_output_qubits: int,
                 func_dict: Dict[Union[int, Tuple[int, int]], int]) -> None:
        """
        Args:
            operation_counts: The counts of each operation performed per iteration.
            n_input_qubits: The number of qubits used to represent the input.
            n_output_qubits: The number of qubits used to represent the output.
            func_dict: A dictionary representation of the function, where the keys correspond
                to a variable, and the values are the corresponding coefficients.
        """
        self._operation_counts = operation_counts
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._func_dict = func_dict

    @property
    def operation_counts(self) -> Dict[int, Dict[str, int]]:
        """Get the operation counts.
        
        Returns:
            The counts of each operation performed per iteration.
        """
        return self._operation_counts

    @property
    def n_input_qubits(self) -> int:
        """Getter of n_input_qubits

        Returns:
            The number of qubits used to represent the input.
        """
        return self._n_input_qubits

    @property
    def n_output_qubits(self) -> int:
        """Getter of n_output_qubits

        Returns:
            The number of qubits used to represent the output.
        """
        return self._n_output_qubits

    @property
    def func_dict(self) -> Dict[Union[int, Tuple[int, int]], int]:
        """Getter of func_dict

        Returns:
            A dictionary of coefficients describing a function, where the keys are the subscripts
            of the variables (e.g. x1), and the values are the corresponding coefficients. If there
            is a constant term, it is referenced by key -1.
        """
        return self._func_dict

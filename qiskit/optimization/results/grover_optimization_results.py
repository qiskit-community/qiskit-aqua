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

from qiskit.optimization.results import OptimizationResult


class GroverOptimizationResults(OptimizationResult):

    """A results object for Grover Optimization methods."""

    def __init__(self, optimum_input, optimum_output, operation_counts, rotations, n_input_qubits,
                 n_output_qubits, func_dict):
        """
        Constructor.

        Args:
            optimum_input (int): The input that corresponds to the optimum output.
            optimum_output (int): The optimum output value.
            operation_counts (dict): The counts of each operation performed per iteration.
            rotations (int): The total number of Grover rotations performed.
            n_input_qubits (int): The number of qubits used to represent the input.
            n_output_qubits (int): The number of qubits used to represent the output.
            func_dict (dict): A dictionary representation of the function, where the keys correspond
                to a variable, and the values are the corresponding coefficients.
        """
        super().__init__(optimum_input, optimum_output)
        self._optimum_input = optimum_input
        self._optimum_output = optimum_output
        self._operation_counts = operation_counts
        self._rotations = rotations
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._func_dict = func_dict

    @property
    def optimum_input(self):
        """Getter of optimum_input"""
        return self._optimum_input

    @property
    def optimum_output(self):
        """Getter of optimum_output"""
        return self._optimum_output

    @property
    def operation_counts(self):
        """Getter of operation_counts"""
        return self._operation_counts

    @property
    def rotation_count(self):
        """Getter of rotation_count"""
        return self._rotations

    @property
    def n_input_qubits(self):
        """Getter of n_input_qubits"""
        return self._n_input_qubits

    @property
    def n_output_qubits(self):
        """Getter of n_output_qubits"""
        return self._n_output_qubits

    @property
    def func_dict(self):
        """Getter of func_dict"""
        return self._func_dict

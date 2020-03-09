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


class GroverOptimizationResults:

    def __init__(self, optimum_input, optimum_output, operation_counts, rotations, n_input_qubits, n_output_qubits, f):
        self._optimum_input = optimum_input
        self._optimum_output = optimum_output
        self._operation_counts = operation_counts
        self._rotations = rotations
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._f = f

    @property
    def optimum_input(self):
        return self._optimum_input

    @property
    def optimum_output(self):
        return self._optimum_output

    @property
    def operation_counts(self):
        return self._operation_counts

    @property
    def rotation_count(self):
        return self._rotations

    @property
    def n_input_qubits(self):
        return self._n_input_qubits

    @property
    def n_output_qubits(self):
        return self._n_output_qubits

    @property
    def function(self):
        return self._f

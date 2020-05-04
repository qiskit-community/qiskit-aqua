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


"""The converter from a ```Operator``` to ``QuadraticProgram``."""

import copy

import numpy as np

from qiskit.aqua.operators.legacy import WeightedPauliOperator
from ..problems.quadratic_program import QuadraticProgram
from ..exceptions import QiskitOptimizationError


class IsingToQuadraticProgram:
    """Convert a qubit operator into a quadratic program"""

    def __init__(self) -> None:
        """Initialize the internal data structure."""
        self._qubit_op = None
        self._offset = 0
        self._num_qubits = 0
        self._qubo_matrix = None
        self._qp = None

    def encode(self, qubit_op: WeightedPauliOperator, offset: float = 0.0) -> QuadraticProgram:
        """Convert a qubit operator and a shift value into a quadratic program

        Args:
            qubit_op: The qubit operator to be converted into a
                :class:`~qiskit.optimization.problems.quadratic_program.QuadraticProgram`
            offset: The shift value of the qubit operator

        Returns:
            QuadraticProgram converted from the input qubit operator and the shift value

        Raises:
            QiskitOptimizationError: If there are Pauli Xs in any Pauli term
            QiskitOptimizationError: If there are more than 2 Pauli Zs in any Pauli term
        """
        # Set properties
        self._qubit_op = qubit_op
        self._offset = copy.deepcopy(offset)
        self._num_qubits = qubit_op.num_qubits

        # Create `QuadraticProgram`
        self._qp = QuadraticProgram()
        for i in range(self._num_qubits):
            self._qp.binary_var(name='x_{0}'.format(i))
        # Create QUBO matrix
        self._create_qubo_matrix()

        # Initialize dicts for linear terms and quadratic terms
        linear = {}
        quadratic = {}

        # For quadratic pauli terms of operator
        # x_i * x_ j = (1 - Z_i - Z_j + Z_i * Z_j)/4
        for i, row in enumerate(self._qubo_matrix):
            for j, weight in enumerate(row):
                # Focus on the upper triangular matrix
                if j <= i:
                    continue
                # Add a quadratic term to the object function of `QuadraticProgram`
                # The coefficient of the quadratic term in `QuadraticProgram` is
                # 4 * weight of the pauli
                coef = weight * 4
                quadratic[(i, j)] = coef
                # Sub the weight of the quadratic pauli term from the QUBO matrix
                self._qubo_matrix[i, j] -= weight
                # Sub the weight of the linear pauli term from the QUBO matrix
                self._qubo_matrix[i, i] += weight
                self._qubo_matrix[j, j] += weight
                # Sub the weight from offset
                offset -= weight

        # After processing quadratic pauli terms, only linear paulis are left
        # x_i = (1 - Z_i)/2
        for i in range(self._num_qubits):
            weight = self._qubo_matrix[i, i]
            # Add a linear term to the object function of `QuadraticProgram`
            # The coefficient of the linear term in `QuadraticProgram` is
            # 2 * weight of the pauli
            coef = weight * 2
            linear[i] = -coef
            # Sub the weight of the linear pauli term from the QUBO matrix
            self._qubo_matrix[i, i] -= weight
            offset += weight

        self._qp.minimize(offset, linear, quadratic)
        offset -= offset

        return self._qp

    def _create_qubo_matrix(self):
        """Create a QUBO matrix from the qubit operator

        Raises:
            QiskitOptimizationError: If there are Pauli Xs in any Pauli term
            QiskitOptimizationError: If there are more than 2 Pauli Zs in any Pauli term

        """
        # Set properties
        # The Qubo matrix is an upper triangular matrix.
        # Diagonal elements in the QUBO matrix is for linear terms of the qubit operator
        # The other elements in the QUBO matrix is for quadratic terms of the qubit operator
        self._qubo_matrix = np.zeros((self._num_qubits, self._num_qubits))

        for pauli in self._qubit_op.paulis:
            # Count the number of Pauli Zs in a Pauli term
            lst_z = pauli[1].z.tolist()
            z_index = [i for i, z in enumerate(lst_z) if z is True]
            num_z = len(z_index)

            # Add its weight of the Pauli term to the corresponding element of QUBO matrix
            if num_z == 1:
                self._qubo_matrix[z_index[0], z_index[0]] = pauli[0].real
            elif num_z == 2:
                self._qubo_matrix[z_index[0], z_index[1]] = pauli[0].real
            else:
                raise QiskitOptimizationError(
                    'There are more than 2 Pauli Zs in the Pauli term {}'.format(pauli[1].z)
                )

            # If there are Pauli Xs in the Pauli term, raise an error
            lst_x = pauli[1].x.tolist()
            x_index = [i for i, x in enumerate(lst_x) if x is True]
            if len(x_index) > 0:
                raise QiskitOptimizationError('Pauli Xs exist in the Pauli {}'.format(pauli[1].x))

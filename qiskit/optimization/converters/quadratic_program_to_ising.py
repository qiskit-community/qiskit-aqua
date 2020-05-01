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


"""The converter from an ```QuadraticProgram``` to ``Operator``."""

from typing import Tuple

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import WeightedPauliOperator

from ..problems.quadratic_program import QuadraticProgram
from ..exceptions import QiskitOptimizationError


class QuadraticProgramToIsing:
    """Convert an optimization problem into a qubit operator."""

    def __init__(self) -> None:
        """Initialize the internal data structure."""
        self._src = None

    def encode(self, op: QuadraticProgram) -> Tuple[WeightedPauliOperator, float]:
        """Convert a problem into a qubit operator

        Args:
            op: The optimization problem to be converted. Must be an unconstrained problem with
                binary variables only.

        Returns:
            The qubit operator of the problem and the shift value.

        Raises:
            QiskitOptimizationError: If a variable type is not binary.
            QiskitOptimizationError: If constraints exist in the problem.
        """

        self._src = op
        # if op has variables that are not binary, raise an error
        if self._src.get_num_vars() > self._src.get_num_binary_vars():
            raise QiskitOptimizationError('The type of variable must be a binary variable.')

        # if constraints exist, raise an error
        if self._src.linear_constraints \
                or self._src.quadratic_constraints:
            raise QiskitOptimizationError('An constraint exists. '
                                          'The method supports only model with no constraints.')

        # initialize Hamiltonian.
        num_nodes = self._src.get_num_vars()
        pauli_list = []
        shift = 0
        zero = np.zeros(num_nodes, dtype=np.bool)

        # set a sign corresponding to a maximized or minimized problem.
        # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
        sense = self._src.objective.sense.value

        # convert a constant part of the object function into Hamiltonian.
        shift += self._src.objective.constant * sense

        # convert linear parts of the object function into Hamiltonian.
        for i, coef in self._src.objective.linear.to_dict().items():
            z_p = np.zeros(num_nodes, dtype=np.bool)
            weight = coef * sense / 2
            z_p[i] = True

            pauli_list.append([-weight, Pauli(z_p, zero)])
            shift += weight

        # convert quadratic parts of the object function into Hamiltonian.
        # first merge coefficients (i, j) and (j, i)
        coeffs = {}
        for (i, j), coeff in self._src.objective.quadratic.to_dict().items():
            if j < i:
                coeffs[(j, i)] = coeffs.get((j, i), 0.0) + coeff
            else:
                coeffs[(i, j)] = coeffs.get((i, j), 0.0) + coeff

        # create Pauli terms
        for (i, j), coeff in coeffs.items():

            weight = coeff * sense / 4

            if i == j:
                shift += weight
            else:
                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([weight, Pauli(z_p, zero)])

            z_p = np.zeros(num_nodes, dtype=np.bool)
            z_p[i] = True
            pauli_list.append([-weight, Pauli(z_p, zero)])

            z_p = np.zeros(num_nodes, dtype=np.bool)
            z_p[j] = True
            pauli_list.append([-weight, Pauli(z_p, zero)])

            shift += weight

        # Remove paulis whose coefficients are zeros.
        qubit_op = WeightedPauliOperator(paulis=pauli_list)

        return qubit_op, shift

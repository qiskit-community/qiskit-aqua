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

from typing import Optional, Union
import copy
import warnings
import numpy as np  # pylint: disable=unused-import

from qiskit.aqua.operators import OperatorBase, WeightedPauliOperator
from ..problems.quadratic_program import QuadraticProgram


class IsingToQuadraticProgram:
    """Convert a qubit operator into a quadratic program"""

    def __init__(self, linear: bool = False) -> None:
        r"""
        Args:
            linear: If linear is True, :math:`x^2` is treated as a linear term
                since :math:`x^2 = x` for :math:`x \in \{0,1\}`.
                Else, :math:`x^2` is treat as a quadratic term.
                The default value is False.
        """
        self._qubit_op = None
        self._offset = 0.0
        self._num_qubits = 0
        self._qubo_matrix = None  # type: Optional[np.ndarray]
        self._qp = None  # type: Optional[QuadraticProgram]
        self._linear = linear
        warnings.warn("The IsingToQuadraticProgram class is deprecated and "
                      "will be removed in a future release. Use the "
                      ".from_ising() method on the QuadraticProgram class "
                      "instead.", DeprecationWarning)

    def encode(self, qubit_op: Union[OperatorBase, WeightedPauliOperator], offset: float = 0.0
               ) -> QuadraticProgram:
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
            NotImplementedError: If the input operator is a ListOp
        """
        self._qubit_op = qubit_op
        self._offset = copy.deepcopy(offset)
        self._num_qubits = qubit_op.num_qubits
        self._qp = QuadraticProgram()
        self._qp.from_ising(qubit_op, offset,
                            linear=self._linear)
        return self._qp

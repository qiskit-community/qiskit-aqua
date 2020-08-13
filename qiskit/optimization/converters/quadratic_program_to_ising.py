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

from typing import Tuple, Optional
import warnings

from qiskit.aqua.operators import OperatorBase
from ..problems.quadratic_program import QuadraticProgram


class QuadraticProgramToIsing:
    """Convert an optimization problem into a qubit operator."""

    def __init__(self) -> None:
        """Initialize the internal data structure."""
        self._src = None  # type: Optional[QuadraticProgram]
        warnings.warn("The QuadraticProgramToIsing class is deprecated and "
                      "will be removed in a future release. Use the "
                      ".to_ising() method on a QuadraticProgram object "
                      "instead.", DeprecationWarning)

    def encode(self, op: QuadraticProgram) -> Tuple[OperatorBase, float]:
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
        return self._src.to_ising()

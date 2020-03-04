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

"""The Exact Minimum Eigensolver algorithm."""

from typing import List, Optional

from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.operators import LegacyBaseOperator


# pylint: disable=invalid-name

class ExactMinimumEigensolver(ExactEigensolver):
    """
    The Exact Minimum Eigensolver algorithm.
    """

    def __init__(self, operator: LegacyBaseOperator,
                 aux_operators: Optional[List[LegacyBaseOperator]] = None) -> None:
        """
        Args:
            operator: Operator instance
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
        """
        super().__init__(operator, 1, aux_operators)

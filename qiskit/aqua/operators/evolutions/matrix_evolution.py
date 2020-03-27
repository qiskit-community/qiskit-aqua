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

""" Expectation Algorithm Base """

import logging

from ..operator_base import OperatorBase
from .evolution_base import EvolutionBase

logger = logging.getLogger(__name__)


class MatrixEvolution(EvolutionBase):
    """ TODO - blocked on whether we can make the UnitaryGate hold the matrix and a
    ParameterExpression for the evolution time.

    """

    def __init__(self):
        """
        Args:

        """
        pass

    def convert(self, operator: OperatorBase) -> OperatorBase:
        pass

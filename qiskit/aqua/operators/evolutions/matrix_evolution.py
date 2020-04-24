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
from .evolved_op import EvolvedOp
from ..primitive_ops.pauli_op import PauliOp
from ..primitive_ops.matrix_op import MatrixOp
from ..list_ops.list_op import ListOp

logger = logging.getLogger(__name__)


class MatrixEvolution(EvolutionBase):
    """ Constructs a circuit with Unitary or HamiltonianGates to represent the exponentiation of
    the operator.

    """

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if isinstance(operator, EvolvedOp):
            if isinstance(operator.primitive, ListOp):
                return operator.primitive.to_matrix_op().exp_i() * operator.coeff
            elif isinstance(operator.primitive, (MatrixOp, PauliOp)):
                return operator.primitive.exp_i()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()

        return operator

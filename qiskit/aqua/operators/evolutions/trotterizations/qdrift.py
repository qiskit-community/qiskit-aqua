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

"""
QDrift Class

"""

import numpy as np

from .trotterization_base import TrotterizationBase
from ...list_ops.summed_op import SummedOp
from ...list_ops.composed_op import ComposedOp


# pylint: disable=invalid-name

class QDrift(TrotterizationBase):
    """ The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in https://arxiv.org/abs/1811.08017.
    """

    def __init__(self, reps: int = 1) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
        """
        super().__init__(reps=reps)

    # pylint: disable=arguments-differ
    def convert(self, op_sum: SummedOp) -> ComposedOp:
        # We artificially make the weights positive, TODO check if this works
        weights = np.abs([op.coeff for op in op_sum.oplist])
        lambd = sum(weights)
        N = 2 * (lambd ** 2) * (op_sum.coeff ** 2)

        factor = lambd * op_sum.coeff / (N * self.reps)
        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant factor.
        scaled_ops = [(op * (factor / op.coeff)).exp_i() for op in op_sum.oplist]
        sampled_ops = np.random.choice(scaled_ops, size=(int(N * self.reps),), p=weights / lambd)

        return ComposedOp(sampled_ops).reduce()

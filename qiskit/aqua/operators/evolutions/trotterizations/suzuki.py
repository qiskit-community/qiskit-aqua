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
Simple Trotter expansion.

"""

from typing import List, Union
from qiskit.quantum_info import Pauli

from .trotterization_base import TrotterizationBase
from ...combo_operators import ComposedOp, SummedOp


class Suzuki(TrotterizationBase):
    """ Simple Trotter expansion """
    def __init__(self,
                 reps: int = 1,
                 order: int = 2) -> None:
        super().__init__(reps=reps)
        self._order = order

    @property
    def order(self) -> int:
        """ returns order """
        return self._order

    @order.setter
    def order(self, order: int) -> None:
        """ sets order """
        self._order = order

    def trotterize(self, op_sum: SummedOp) -> ComposedOp:
        composition_list = Suzuki.suzuki_recursive_expansion(
            op_sum.oplist, op_sum.coeff, self.order, self.reps)

        single_rep = ComposedOp(composition_list)
        full_evo = single_rep.power(self.reps)
        return full_evo.reduce()

    @staticmethod
    def suzuki_recursive_expansion(op_list: List[List[Union[complex, Pauli]]],
                                   evo_time: float,
                                   expansion_order: int,
                                   reps: int) -> List:
        """
        Compute the list of pauli terms for a single slice of the suzuki expansion
        following the paper https://arxiv.org/pdf/quant-ph/0508139.pdf.

        Args:
            op_list: The slice's weighted Pauli list for the suzuki expansion
            evo_time: The parameter lambda as defined in said paper,
                              adjusted for the evolution time and the number of time slices
            expansion_order: The order for suzuki expansion
            reps: reps
        Returns:
            list: slice pauli list
        """
        if expansion_order == 1:
            # Base first-order Trotter case
            return [(op * (evo_time / reps)).exp_i() for op in op_list]
        if expansion_order == 2:
            half = Suzuki.suzuki_recursive_expansion(op_list, evo_time / 2,
                                                     expansion_order - 1, reps)
            return list(reversed(half)) + half
        else:
            p_k = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
            side = 2 * Suzuki.suzuki_recursive_expansion(op_list, evo_time
                                                         * p_k, expansion_order - 2, reps)
            middle = Suzuki.suzuki_recursive_expansion(op_list, evo_time * (1 - 4 * p_k),
                                                       expansion_order - 2, reps)
            return side + middle + side

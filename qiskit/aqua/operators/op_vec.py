# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Eager Operator Vec Container """

from .op_combo_base import OpCombo


class OpVec(OpCombo):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
            coeff (float, complex): A coefficient multiplying the primitive

            Note that the "recombination function" below is the identity - it takes a list of operators,
            and is supposed to return a list of operators.
        """
        super().__init__(oplist, combo_fn=lambda x: x, coeff=coeff)

    # For now, follow tensor convention that each Operator in the vec is a separate "system"
    @property
    def num_qubits(self):
        return sum([op.num_qubits for op in self.oplist])

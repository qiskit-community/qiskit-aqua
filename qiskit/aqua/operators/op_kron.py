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

""" Eager Operator Kron Container """

from .op_combo_base import OpCombo
from functools import reduce, partial


class OpKron(OpCombo):

    def __init__(self, oplist, coeff=1.0):
        """
        Args:
            oplist (list(OperatorBase)): The operators being summed.
            coeff (float, complex): A coefficient multiplying the primitive
        """
        super().__init__(oplist, combo_fn=partial(reduce, np.kron), coeff=coeff)

    @property
    def num_qubits(self):
        return sum([op.num_qubits for op in self.oplist])

    def kron(self, other):
        """ Kron """
        if isinstance(other, OpKron):
            return OpKron(self.oplist + other.oplist, coeff=self.coeff * other.coeff)
        return OpKron(self.oplist + [other], coeff=self.coeff)

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

"""The module for Quantum the Fisher Information."""
from collections.abc import Iterable
from typing import List, Union, Optional

from .qfi_base import QFIBase
from qiskit.aqua.operators import OperatorBase, ListOp
from qiskit.aqua.operators.gradients import CircuitQFI
from qiskit.aqua.operators.state_fns import CircuitStateFn
from qiskit.circuit import (ParameterExpression, ParameterVector)

from qiskit.aqua.operators.expectations import PauliExpectation


class QFI(QFIBase):
    r"""Compute the Quantum Fisher Information (QFI) given a pure, parametrized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def __init__(self,
                 qfi_method: Union[str, CircuitQFI] = 'lin_comb_full'):
        r"""
        Args:
            qfi_method: The method used to compute the state/probability gradient. Can be either
                ``'lin_comb_full'`` or ``'overlap_diag'``` or ``'overlap_block_diag'```.
        """
        super().__init__(qfi_method)

    def convert(self,
                operator: CircuitStateFn,
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression]]] = None
                ) -> ListOp(List[OperatorBase]):
        r"""
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl

        Raises:
            ValueError: If the value for ``approx`` is not supported.
        """
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)

        return self.qfi_method.convert(cleaned_op, params)

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

from qiskit.aqua.operators import OperatorBase, ListOp
from qiskit.aqua.operators.gradients import DerivativeBase, CircuitGradientMethod
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import (Parameter, ParameterVector)

from qiskit.aqua.operators.expectations import PauliExpectation


class QFI(DerivativeBase):
    r"""Compute the Quantum Fisher Information (QFI) given a pure, parametrized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def __init__(self,
                 method: Union[str, CircuitGradientMethod] = 'lin_comb',
                 **kwargs):
        r"""
        Args:
            method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
                Deprecated for observable gradient.
            epsilon: The offset size to use when computing finite difference gradients.


        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        if isinstance(method, CircuitGradientMethod):
            self._method = method

        elif method == 'lin_comb':
            from .circuit_gradient_methods import LinCombQFI
            self._method = LinCombQFI()
        elif method == 'block_diag':
            from .circuit_gradient_methods import BlockDiagQFI
            self._method = BlockDiagQFI()
        elif method == 'diag':
            from .circuit_gradient_methods import DiagQFI
            self._method = DiagQFI()
        else:
            raise ValueError("Unrecognized input provided for `method`. Please provide"
                             " a CircuitGradientMethod object or one of the pre-defined string"
                             " arguments: {'lin_comb', 'diag', 'block_diag'}. ")

    def convert(self,
                operator: CircuitStateFn,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
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

        return self._method.convert(cleaned_op, params)

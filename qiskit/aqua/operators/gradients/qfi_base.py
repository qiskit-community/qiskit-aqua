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

from typing import Union

from qiskit.aqua.operators.gradients import DerivativeBase, CircuitGradientMethod


class QFIBase(DerivativeBase):
    r"""Compute the Quantum Fisher Information (QFI) given a pure, parametrized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def __init__(self,
                 qfi_method: Union[str, CircuitGradientMethod] = 'lin_comb'):
        r"""
        Args:
            qfi_method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
                Deprecated for observable gradient.


        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        if isinstance(qfi_method, CircuitGradientMethod):
            self._qfi_method = qfi_method

        elif qfi_method == 'lin_comb':
            from .circuit_gradient_methods import LinCombQFI
            self._qfi_method = LinCombQFI()
        elif qfi_method == 'block_diag':
            from .circuit_gradient_methods import BlockDiagQFI
            self._qfi_method = BlockDiagQFI()
        elif qfi_method == 'diag':
            from .circuit_gradient_methods import DiagQFI
            self._qfi_method = DiagQFI()
        else:
            raise ValueError("Unrecognized input provided for `method`. Please provide"
                             " a CircuitGradientMethod object or one of the pre-defined string"
                             " arguments: {'lin_comb', 'diag', 'block_diag'}. ")

    @property
    def qfi_method(self):
        return self._qfi_method

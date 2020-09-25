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

"""The base interface for Aqua's gradient."""
from typing import Union

from qiskit.aqua.operators.gradients.circuit_gradients import CircuitGradient

from qiskit.aqua.operators.gradients.derivatives_base import DerivativeBase


class GradientBase(DerivativeBase):
    """Convert an operator expression to the first-order gradient."""

    def __init__(self,
                 grad_method: Union[str, CircuitGradient] = 'param_shift',
                 **kwargs):
        r"""
        Args:
            grad_method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``. Deprecated for observable
                gradient.
            epsilon: The offset size to use when computing finite difference gradients.


        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        if isinstance(grad_method, CircuitGradient):
            self._grad_method = grad_method
        elif grad_method == 'param_shift':
            from .circuit_gradients.param_shift import ParamShift
            self._grad_method = ParamShift(analytic=True)

        elif grad_method == 'fin_diff':
            from .circuit_gradients.param_shift import ParamShift
            if 'epsilon' in kwargs:
                epsilon = kwargs['epsilon']
            else:
                epsilon = 1e-6
            self._grad_method = ParamShift(analytic=False, epsilon=epsilon)

        elif grad_method == 'lin_comb':
            from .circuit_gradients.lin_comb import LinComb
            self._grad_method = LinComb()
        else:
            raise ValueError("Unrecognized input provided for `method`. Please provide"
                             " a CircuitGradientMethod object or one of the pre-defined string"
                             " arguments: {'param_shift', 'fin_diff', 'lin_comb'}. ")

    @property
    def grad_method(self):
        return self._grad_method

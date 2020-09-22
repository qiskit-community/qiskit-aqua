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

"""The base interface for Aqua's gradient."""
from typing import Union, List, Optional

import numpy as np
from jax import grad, jit
from qiskit.aqua import AquaError
from qiskit.aqua.operators import (
    PauliExpectation
)
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector

from qiskit.aqua.operators.gradients.derivatives_base import DerivativeBase
from qiskit.aqua.operators.gradients.circuit_gradient_methods.circuit_gradient_method import CircuitGradientMethod
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.summed_op import SummedOp
from qiskit.aqua.operators.list_ops.tensored_op import TensoredOp
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.operator_globals import Zero, One
from qiskit.aqua.operators.state_fns.circuit_state_fn import CircuitStateFn


class Gradient(DerivativeBase):
    """Convert an operator expression to the first-order gradient."""

    def __init__(self,
                 method: Union[str, CircuitGradientMethod] = 'param_shift',
                 **kwargs):
        r"""
        Args:
            method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``. Deprecated for observable gradient.
            epsilon: The offset size to use when computing finite difference gradients.

        
        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """

        if isinstance(method, CircuitGradientMethod):
            self._method = method
        elif method == 'param_shift':
            from .circuit_gradient_methods.param_shift_gradient import ParamShiftGradient
            self._method = ParamShiftGradient(analytic=True)

        elif method == 'fin_diff':
            from .circuit_gradient_methods.param_shift_gradient import ParamShiftGradient
            if 'epsilon' in kwargs:
                epsilon = kwargs['epsilon']
            else:
                epsilon = 1e-6
            self._method = ParamShiftGradient(analytic=False, epsilon=epsilon)

        elif method == 'lin_comb':
            from .circuit_gradient_methods.lin_comb_gradient import LinCombGradient
            self._method = LinCombGradient()
        else:
            raise ValueError("Unrecognized input provided for `method`. Please provide" 
                             " a CircuitGradientMethod object or one of the pre-defined string" 
                             " arguments: {'param_shift', 'fin_diff', 'lin_comb'}. ")

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, Parameter, List[Parameter]]] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            params: params: The parameters we are taking the gradient with respect to.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        # grad_combo_fn: Gradient for a custom operator combo_fn. The gradient for a standard
        #     ``ListOp`` or SymPy combo_fn is automatically computed.

        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        # TODO where is the param = None case handled?
        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.convert(operator, param) for param in params]
            # If autograd returns None, then the corresponding parameter was probably not present
            # in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params
                )
            return ListOp(param_grads)

        param = params

        # Preprocessing
        # TODO think of better names...
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_gradient(cleaned_op, param)

    @property
    def method(self):
        return self._method

    def get_gradient(self,
                     operator: OperatorBase,
                     params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]]
                     ) -> OperatorBase:
        """TODO

        Args:
            operator: TODO
            params: TODO
            method: TODO

        Returns:
            TODO

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            AquaError: If the coefficent of the operator could not be reduced to 1.
            NotImplementedError: If operator is a TensoredOp  # TODO support this
            Exception: Unintended code is reached  # TODO proper warnings and errors
        """

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.get_gradient(operator, param) for param in params]
            # If autograd returns None, then the corresponding parameter was probably not present
            # in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    'The following parameters do not appear in the provided operator: ',
                    absent_params
                )
            return ListOp(param_grads)

        # by this point, it's only one parameter
        param = params
        # Handle Product Rules
        # print('Op coeff ', operator._coeff)
        if not is_coeff_c(operator._coeff, 1.0):

            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator / coeff
            # Get derivative of the operator (recursively)
            d_op = self.get_gradient(op, param)
            # ..get derivative of the coeff
            d_coeff = self.parameter_expression_grad(coeff, param)

            grad_op = 0
            if d_op != ~Zero @ One and not is_coeff_c(coeff, 0.0):
                grad_op += coeff * d_op
            if op != ~Zero @ One and not is_coeff_c(d_coeff, 0.0):
                grad_op += d_coeff * op
            if grad_op == 0:
                grad_op = ~Zero @ One
            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):

            # Gradient of an expectation value
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO if this is a sum of circuit state fns - traverse including autograd
            if isinstance(operator[-1], (CircuitStateFn)):
                pass
                # Do some checks and decide how you're planning on taking the gradient.
                # for now we do param shift
            else:
                raise TypeError(
                    'The gradient framework is compatible with states that are given as CircuitStateFn')

            return self.method.convert(operator, param)

        elif isinstance(operator, CircuitStateFn):
            # Gradient of an a state's sampling probabilities
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')
            return self.method.convert(operator, param)

        # Handle the chain rule
        elif isinstance(operator, ListOp):
            grad_ops = [self.get_gradient(op, param) for op in operator.oplist]

            # Note: this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn.
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and raise an error.
            # An alternative is to check the byte code of the operator's combo_fn against the
            # default one.
            # This will work but look very ugly and may have other downsides I'm not aware of
            if operator._combo_fn == ListOp([])._combo_fn:
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
                return SummedOp(oplist=grad_ops)
            elif isinstance(operator, TensoredOp):
                return TensoredOp(oplist=grad_ops)

            if operator.grad_combo_fn:
                grad_combo_fn = operator.grad_combo_fn
            else:
                try:
                    grad_combo_fn = jit(grad(operator._combo_fn, holomorphic=True))
                except Exception:
                    raise TypeError(
                        'This automatic differentiation function is based on JAX. Please use import '
                        'jax.numpy as jnp instead of import numpy as np when defining a combo_fn.')

            # f(g_1(x), g_2(x)) --> df/dx = df/dg_1 dg_1/dx + df/dg_2 dg_2/dx
            return ListOp([ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                          combo_fn=lambda x: np.dot(x[0], x[1]))

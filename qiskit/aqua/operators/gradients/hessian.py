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

"""The module to compute Hessians."""

from typing import Optional, Union, List, Tuple

import numpy as np
from qiskit.aqua.aqua_globals import AquaError
from qiskit.aqua.operators import Zero, One, CircuitStateFn, StateFn
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.gradients.gradient import Gradient
from qiskit.aqua.operators.gradients.hessian_base import HessianBase
from qiskit.aqua.operators.list_ops import ListOp, ComposedOp, SummedOp, TensoredOp
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.circuit import ParameterVector, ParameterExpression

try:
    from jax import grad, jit

    _HAS_JAX_ = True
except ModuleNotFoundError:
    _HAS_JAX_ = False


class Hessian(HessianBase):
    """Compute the Hessian of an expected value."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Tuple[ParameterExpression, ParameterExpression],
                                       List[Tuple[ParameterExpression, ParameterExpression]],
                                       List[ParameterExpression], ParameterVector]] = None
                ) -> OperatorBase:
        """
        Args:
            operator: The operator for which we compute the Hessian
            params: The parameters we are computing the Hessian with respect to
                    Either give directly the tuples/list of tuples for which the second order
                    derivative is to be computed or give a list of parameters to build the
                    full Hessian for those parameters.

        Returns:
            OperatorBase: An operator whose evaluation yields the Hessian

        Raises:
            ValueError: If `params` is not set.
        """
        # if input is a tuple instead of a list, wrap it into a list
        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        if isinstance(params, (ParameterVector, list)):
            # Case: a list of parameters were given, compute the Hessian for all param pairs
            if all(isinstance(param, ParameterExpression) for param in params):
                return ListOp(
                    [ListOp([self.convert(operator, (p0, p1)) for p1 in params]) for p0 in params])
            # Case: a list was given containing tuples of parameter pairs.
            # Compute the Hessian entries corresponding to these pairs of parameters.
            elif all(isinstance(param, tuple) for param in params):
                return ListOp([self.convert(operator, param_pair) for param_pair in params])

        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_hessian(cleaned_op, params)

    # pylint: disable=too-many-return-statements
    def get_hessian(self,
                    operator: OperatorBase,
                    params: Optional[Union[Tuple[ParameterExpression, ParameterExpression],
                                           List[Tuple[
                                               ParameterExpression, ParameterExpression]]]] = None
                    ) -> OperatorBase:
        """Get the Hessian for the given operator w.r.t. the given parameters

        Args:
            operator: Operator w.r.t. which we take the Hessian.
            params: Parameters w.r.t. which we compute the Hessian.

        Returns:
            Operator which represents the gradient w.r.t. the given params.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            AquaError: If the coefficent of the operator could not be reduced to 1.
            NotImplementedError: If operator is a TensoredOp  # TODO support this
            TypeError: TODO
            Exception: Unintended code is reached
        """

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        if isinstance(params, (ParameterVector, list)):
            # Case: a list of parameters were given, compute the Hessian for all param pairs
            if all(isinstance(param, ParameterExpression) for param in params):
                return ListOp(
                    [ListOp([self.get_hessian(operator, (p0, p1)) for p1 in params])
                     for p0 in params])
            # Case: a list was given containing tuples of parameter pairs.
            # Compute the Hessian entries corresponding to these pairs of parameters.
            elif all(isinstance(param, tuple) for param in params):
                return ListOp(
                    [self.get_hessian(operator, param_pair) for param_pair in params])

        # If a gradient is requested w.r.t a single parameter, then call the
        # Gradient() class' autograd method.
        if isinstance(params, ParameterExpression):
            return Gradient(grad_method=self._hess_method).get_gradient(operator, params)

        assert isinstance(params, tuple) and len(
            params) == 2, "Parameters supplied in unsupported format"

        # By this point, it's only one parameter tuple
        p_0 = params[0]
        p_1 = params[1]

        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0):
            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator / coeff
            # Get derivative of the operator (recursively)
            d0_op = self.get_hessian(op, p_0)
            d1_op = self.get_hessian(op, p_1)
            # ..get derivative of the coeff
            d0_coeff = self.parameter_expression_grad(coeff, p_0)
            d1_coeff = self.parameter_expression_grad(coeff, p_1)

            dd_op = self.get_hessian(op, params)
            dd_coeff = self.parameter_expression_grad(d0_coeff, p_1)

            grad_op = 0
            # Avoid creating operators that will evaluate to zero
            if dd_op != ~Zero @ One and not is_coeff_c(coeff, 0):
                grad_op += coeff * dd_op
            if d0_op != ~Zero @ One and not is_coeff_c(d1_coeff, 0):
                grad_op += d1_coeff * d0_op
            if d1_op != ~Zero @ One and not is_coeff_c(d0_coeff, 0):
                grad_op += d0_coeff * d1_op
            if not is_coeff_c(dd_coeff, 0):
                grad_op += dd_coeff * op

            if grad_op == 0:
                return ~Zero @ One

            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):

            if not is_coeff_c(operator._coeff, 1.):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO if this is a sum of circuit state fns - traverse including autograd
            if isinstance(operator[-1], (CircuitStateFn)):
                pass
            else:
                raise TypeError(
                    'The gradient framework is compatible with states that are given as '
                    'CircuitStateFn')

            return self.hess_method.convert(operator, params)

        # This is the recursive case where the chain rule is handled
        elif isinstance(operator, ListOp):
            grad_ops = [self.get_hessian(op, params) for op in operator.oplist]

            # Note that this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn.
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and fail.
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
                if _HAS_JAX_:
                    grad_combo_fn = jit(grad(operator._combo_fn, holomorphic=True))
                else:
                    raise AquaError(
                        'This automatic differentiation function is based on JAX. Please install '
                        'jax and use `import jax.numpy as jnp` instead of `import numpy as np` when'
                        'defining a combo_fn.')

            # f(g_1(x), g_2(x)) --> df/dx = df/dg_1 dg_1/dx + df/dg_2 dg_2/dx
            return ListOp([ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                          combo_fn=lambda x: np.dot(x[0], x[1]))

        elif isinstance(operator, StateFn):
            if operator._is_measurement:
                raise TypeError('The computation of Hessians is only supported for Operators which '
                                'represent expectation values.')

        else:
            raise TypeError('The computation of Hessians is only supported for Operators which '
                            'represent expectation values.')
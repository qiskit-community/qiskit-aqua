# # -*- coding: utf-8 -*-
#
# # This code is part of Qiskit.
# #
# # (C) Copyright IBM 2020.
# #
# # This code is licensed under the Apache License, Version 2.0. You may
# # obtain a copy of this license in the LICENSE.txt file in the root directory
# # of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# #
# # Any modifications or derivative works of this code must retain this
# # copyright notice, and modified files need to carry a notice indicating
# # that they have been altered from the originals.
#
# """The module to compute the Hessian using the parameter shift method."""
#
# from typing import Union, List, Tuple, Optional
#
# from qiskit.aqua.operators import OperatorBase, ListOp
# from qiskit.circuit import Parameter
#
# from qiskit.aqua.operators.gradients.gradient import ParamShiftGradient
#
#
# class HessianParamShiftGradient(ParamShiftGradient):
#     """TODO"""
#
#     def __init__(self,
#                  analytic: bool = True,
#                  epsilon: float = 1e-6):
#         r"""
#         Args:
#             analytic: If True use the parameter shift rule to compute analytic gradients,
#                       else use a finite difference approach
#             epsilon: The offset size to use when computing finite difference gradients.
#
#
#         Raises:
#             ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
#         """
#
#         super().__init__(analytic=analytic, epsilon=epsilon)
#
#
#     def single_deriv(self):
#         return ParamShiftGradient(analytic=self.analytic, epsilon=self.epsilon)
#
#
#
#     @property
#     def analytic(self):
#         return self._analytic
#
#     @property
#     def epsilon(self):
#         return self._epsilon
#
#     def convert(self,
#                 operator: OperatorBase,
#                 params: Optional[Union[Tuple[Parameter, Parameter], List[Tuple[Parameter, Parameter]]]] = None) -> OperatorBase:
#         r"""
#         Args:
#             operator: The measurement operator we are taking the gradient of
#             params: The parameters we are taking the gradient with respect to
#             analytic: If True use the parameter shift rule to compute analytic gradients,
#                       else use a finite difference approach
#
#         Returns:
#             An operator whose evaluation yeild the Hessian
#         """
#         if isinstance(params, tuple):
#             return self.parameter_shift(self.parameter_shift(operator, params[0]), params[1])
#         return ListOp([self.parameter_shift(self.parameter_shift(operator, pair[0]), pair[1])
#                        for pair in params])

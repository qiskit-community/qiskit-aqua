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

"""The converter to map integer variables in a quadratic program to binary variables."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..algorithms.optimization_algorithm import OptimizationResult
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class IntegerToBinary:
    """Convert a :class:`~qiskit.optimization.problems.QuadraticProgram` into new one by encoding
    integer with binary variables.

    This bounded-coefficient encoding used in this converted is proposed in [1], Eq. (5).

    Examples:
        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> from qiskit.optimization.converters import IntegerToBinary
        >>> problem = QuadraticProgram()
        >>> var = problem.integer_var(name='x', lowerbound=0, upperbound=10)
        >>> conv = IntegerToBinary()
        >>> problem2 = conv.encode(problem)

    References:
        [1]: Sahar Karimi, Pooya Ronagh (2017), Practical Integer-to-Binary Mapping for Quantum
            Annealers. arxiv.org:1706.01945.
    """

    _delimiter = '@'  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        self._src = None
        self._dst = None
        self._conv = {}  # Dict[Variable, List[Tuple[str, int]]]
        # e.g., self._conv = {x: [('x@1', 1), ('x@2', 2)]}

    def encode(self, op: QuadraticProgram, name: Optional[str] = None) -> QuadraticProgram:
        """Convert an integer problem into a new problem with binary variables.

        Args:
            op: The problem to be solved, that may contain integer variables.
            name: The name of the converted problem. If not provided, the name of the input
                problem is used.

        Returns:
            The converted problem, that contains no integer variables.

        Raises:
            QiskitOptimizationError: if variable or constraint type is not supported.
        """

        # copy original QP as reference.
        self._src = copy.deepcopy(op)

        if self._src.get_num_integer_vars() > 0:

            # initialize new QP
            self._dst = QuadraticProgram()

            # declare variables
            for x in self._src.variables:
                if x.vartype == Variable.Type.INTEGER:
                    new_vars = self._encode_var(x.name, x.lowerbound, x.upperbound)
                    self._conv[x] = new_vars
                    for (var_name, _) in new_vars:
                        self._dst.binary_var(var_name)
                else:
                    if x.vartype == Variable.Type.CONTINUOUS:
                        self._dst.continuous_var(x.lowerbound, x.upperbound, x.name)
                    elif x.vartype == Variable.Type.BINARY:
                        self._dst.binary_var(x.name)
                    else:
                        raise QiskitOptimizationError(
                            "Unsupported variable type {}".format(x.vartype))

            self._substitute_int_var()

        else:
            # just copy the problem if no integer variables exist
            self._dst = copy.deepcopy(op)

        # adjust name of resulting problem if necessary
        if name:
            self._dst.name = name
        else:
            self._dst.name = self._src.name

        return self._dst

    def _encode_var(self, name: str, lowerbound: int, upperbound: int) -> List[Tuple[str, int]]:
        var_range = upperbound - lowerbound
        power = int(np.log2(var_range))
        bounded_coef = var_range - (2 ** power - 1)

        coeffs = [2 ** i for i in range(power)] + [bounded_coef]
        return [(name + self._delimiter + str(i), coef) for i, coef in enumerate(coeffs)]

    def _encode_linear_coefficients_dict(self, coefficients: Dict[str, float]) \
            -> Tuple[Dict[str, float], float]:
        constant = 0
        linear = {}
        for name, v in coefficients.items():
            x = self._src.get_variable(name)
            if x in self._conv:
                for y, coeff in self._conv[x]:
                    linear[y] = v * coeff
                constant += v * x.lowerbound
            else:
                linear[x.name] = v

        return linear, constant

    def _encode_quadratic_coefficients_dict(self, coefficients: Dict[Tuple[str, str], float]) \
            -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
        constant = 0
        linear = {}
        quadratic = {}
        for (name_i, name_j), v in coefficients.items():
            x = self._src.get_variable(name_i)
            y = self._src.get_variable(name_j)

            if x in self._conv and y not in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    quadratic[z_x, y.name] = v * coeff_x
                linear[y.name] = linear.get(y.name, 0.0) + v * x.lowerbound

            elif x not in self._conv and y in self._conv:
                for z_y, coeff_y in self._conv[y]:
                    quadratic[x.name, z_y] = v * coeff_y
                linear[x.name] = linear.get(x.name, 0.0) + v * y.lowerbound

            elif x in self._conv and y in self._conv:
                for z_x, coeff_x in self._conv[x]:
                    for z_y, coeff_y in self._conv[y]:
                        quadratic[z_x, z_y] = v * coeff_x * coeff_y

                for z_x, coeff_x in self._conv[x]:
                    linear[z_x] = linear.get(z_x, 0.0) + v * y.lowerbound
                for z_y, coeff_y in self._conv[y]:
                    linear[z_y] = linear.get(z_y, 0.0) + v * x.lowerbound

                constant += v * x.lowerbound * y.lowerbound

            else:
                quadratic[x.name, y.name] = v

        return quadratic, linear, constant

    def _substitute_int_var(self):

        # set objective
        linear, linear_constant = self._encode_linear_coefficients_dict(
            self._src.objective.linear.to_dict(use_name=True))
        quadratic, quadratic_linear, quadratic_constant = \
            self._encode_quadratic_coefficients_dict(
                self._src.objective.quadratic.to_dict(use_name=True))

        constant = self._src.objective.constant + linear_constant + quadratic_constant
        for i, v in quadratic_linear.items():
            linear[i] = linear.get(i, 0) + v

        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic)
        else:
            self._dst.maximize(constant, linear, quadratic)

        # set linear constraints
        for constraint in self._src.linear_constraints:
            linear, constant = self._encode_linear_coefficients_dict(constraint.linear.to_dict())
            self._dst.linear_constraint(linear, constraint.sense,
                                        constraint.rhs - constant, constraint.name)

        # set quadratic constraints
        for constraint in self._src.quadratic_constraints:
            linear, linear_constant = self._encode_linear_coefficients_dict(
                constraint.linear.to_dict())
            quadratic, quadratic_linear, quadratic_constant = \
                self._encode_quadratic_coefficients_dict(constraint.quadratic.to_dict())

            constant = linear_constant + quadratic_constant
            for i, v in quadratic_linear.items():
                linear[i] = linear.get(i, 0) + v

            self._dst.quadratic_constraint(linear, quadratic, constraint.sense,
                                           constraint.rhs - constant, constraint.name)

    def decode(self, result: OptimizationResult) -> OptimizationResult:
        """Convert the encoded problem (binary variables) back to the original (integer variables).

        Args:
            result: The result of the converted problem.

        Returns:
            The result of the original problem.
        """
        vals = result.x
        new_vals = self._decode_var(vals)
        result.x = new_vals
        return result

    def _decode_var(self, vals) -> List[int]:
        # decode integer values
        sol = {x.name: float(vals[i]) for i, x in enumerate(self._dst.variables)}
        new_vals = []
        for x in self._src.variables:
            if x in self._conv:
                new_vals.append(sum(sol[aux] * coef for aux, coef in self._conv[x]) + x.lowerbound)
            else:
                new_vals.append(sol[x.name])
        return new_vals

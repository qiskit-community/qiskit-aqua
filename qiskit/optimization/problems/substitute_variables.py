# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Mixed integer quadratically constrained quadratic program"""

import logging
from collections import defaultdict
from enum import Enum
from math import fsum
from typing import Optional, Tuple, Dict, Union, List

from qiskit.optimization import infinity, QiskitOptimizationError
from qiskit.optimization.problems.constraint import ConstraintSense
from qiskit.optimization.problems.linear_expression import LinearExpression
from qiskit.optimization.problems.quadratic_expression import QuadraticExpression
from qiskit.optimization.problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


class SubstitutionStatus(Enum):
    """Status of `QuadraticProgram.substitute_variables`"""
    success = 1
    infeasible = 2


class SubstituteVariables:
    """A class to substitute variables of an optimization problem with constants for other
    variables"""

    CONST = '__CONSTANT__'

    def __init__(self):
        self._src: QuadraticProgram = None
        self._dst: QuadraticProgram = None
        self._subs = {}

    def substitute_variables(
            self, src: 'QuadraticProgram',
            constants: Optional[Dict[Union[str, int], float]] = None,
            variables: Optional[Dict[Union[str, int], Tuple[Union[str, int], float]]] = None) \
            -> Tuple[QuadraticProgram, SubstitutionStatus]:
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                e.g., {'x': 2} means 'x' is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., {'x': ('y', 2)} means 'x' is substituted with 'y' * 2

        Returns:
            An optimization problem by substituting variables and the status.
            If the resulting problem has no issue, the status is `success`.
            Otherwise, an empty problem and status `infeasible` are returned.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.
                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.

        Example usage:
            >>>
        """

        self._src = src
        self._dst = QuadraticProgram(src.name)
        # do not set problem type, then it detects its type automatically

        self._subs_dict(constants, variables)

        results = [
            self._variables(),
            self._objective(),
            self._linear_constraints(),
            self._quadratic_constraints(),
        ]
        if any(r == SubstitutionStatus.infeasible for r in results):
            ret = SubstitutionStatus.infeasible
        else:
            ret = SubstitutionStatus.success
        return self._dst, ret

    @staticmethod
    def _feasible(sense: ConstraintSense, rhs: float) -> bool:
        """Checks feasibility of the following condition
            0 `sense` rhs
        """
        # I use the following pylint option because `rhs` should come to right
        # pylint: disable=misplaced-comparison-constant
        if sense == ConstraintSense.eq:
            if 0 == rhs:
                return True
        elif sense == ConstraintSense.leq:
            if 0 <= rhs:
                return True
        elif sense == ConstraintSense.geq:
            if 0 >= rhs:
                return True
        return False

    @staticmethod
    def _replace_dict_keys_with_names(op, dic):
        key = []
        val = []
        for k in sorted(dic.keys()):
            key.append(op.variables.get_names(k))
            val.append(dic[k])
        return key, val

    def _subs_dict(self, constants, variables):
        src = self._src

        # guarantee that there is no overlap between variables to be replaced and combine input
        subs: Dict[str, Tuple[str, float]] = {}
        if constants is not None:
            for i, v in constants.items():
                # substitute i <- v
                i_2 = src.get_variable(i).name
                if i_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable twice: {} <- {}'.format(i, v))
                subs[i_2] = (self.CONST, v)

        if variables is not None:
            for i, (j, v) in variables.items():
                if v == 0:
                    raise QiskitOptimizationError(
                        'coefficient must be non-zero: {} {} {}'.format(i, j, v))
                # substitute i <- j * v
                i_2 = src.get_variable(i).name
                j_2 = src.get_variable(j).name
                if i_2 == j_2:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable: {} <- {} {}'.format(i, j, v))
                if i_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable twice: {} <- {} {}'.format(i, j, v))
                if j_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute by variable that gets substituted itself: '
                        '{} <- {} {}'.format(i, j, v))
                subs[i_2] = (j_2, v)

        self._subs = subs

    def _variables(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst
        subs = self._subs

        # copy variables that are not replaced
        for var in src.variables:
            name = var.name
            vartype = var.vartype
            lowerbound = var.lowerbound
            upperbound = var.upperbound
            if name not in subs:
                dst._add_variable(name, lowerbound, upperbound, vartype)

        for i, (j, v) in subs.items():
            lb_i = src.get_variable(i).lowerbound
            ub_i = src.get_variable(i).upperbound
            if j == self.CONST:
                if not lb_i <= v <= ub_i:
                    logger.warning(
                        'Infeasible substitution for variable: %s', i)
                    return SubstitutionStatus.infeasible
            else:
                # substitute i <- j * v
                # lb_i <= i <= ub_i  -->  lb_i / v <= j <= ub_i / v if v > 0
                #                         ub_i / v <= j <= lb_i / v if v < 0
                if v == 0:
                    raise QiskitOptimizationError(
                        'Coefficient of variable substitution should be nonzero: '
                        '{} {} {}'.format(i, j, v))
                if abs(lb_i) < infinity:
                    new_lb_i = lb_i / v
                else:
                    new_lb_i = lb_i if v > 0 else -lb_i
                if abs(ub_i) < infinity:
                    new_ub_i = ub_i / v
                else:
                    new_ub_i = ub_i if v > 0 else -ub_i
                var_j = dst.get_variable(j)
                lb_j = var_j.lowerbound
                ub_j = var_j.upperbound
                if v > 0:
                    var_j.lowerbound = max(lb_j, new_lb_i)
                    var_j.upperbound = min(ub_j, new_ub_i)
                else:
                    var_j.lowerbound = max(lb_j, new_ub_i)
                    var_j.upperbound = min(ub_j, new_lb_i)

        for var in dst.variables:
            if var.lowerbound > var.upperbound:
                logger.warning(
                    'Infeasible lower and upper bound: %s %f %f', var, var.lowerbound,
                    var.upperbound)
                return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

    def _linear_expression(self, lin_expr: LinearExpression) \
            -> Tuple[List[float], LinearExpression]:
        subs = self._subs
        const = []
        lin_dict = defaultdict(float)
        for i, w_i in lin_expr.coefficients_as_dict(use_index=False).items():
            repl_i = subs[i] if i in subs else (i, 1)
            prod = w_i * repl_i[1]
            if repl_i[0] == self.CONST:
                const.append(prod)
            else:
                k = repl_i[0]
                lin_dict[k] += prod
        new_lin = LinearExpression(quadratic_program=self._dst,
                                   coefficients=lin_dict if lin_dict else {})
        return const, new_lin

    def _quadratic_expression(self, quad_expr: QuadraticExpression) \
            -> Tuple[List[float], Optional[LinearExpression], Optional[QuadraticExpression]]:
        subs = self._subs
        const = []
        lin_dict = defaultdict(float)
        quad_dict = defaultdict(float)
        for (i, j), w_ij in quad_expr.coefficients_as_dict().items():
            repl_i = subs[i] if i in subs else (i, 1)
            repl_j = subs[j] if j in subs else (j, 1)
            idx = tuple(x for x, _ in [repl_i, repl_j] if x != self.CONST)
            prod = w_ij * repl_i[1] * repl_j[1]
            if len(idx) == 2:
                quad_dict[idx] += prod
            elif len(idx) == 1:
                lin_dict[idx[0]] += prod
            else:
                const.append(prod)
        new_lin = LinearExpression(quadratic_program=self._dst,
                                   coefficients=lin_dict if lin_dict else {})
        new_quad = QuadraticExpression(quadratic_program=self._dst,
                                       coefficients=quad_dict if quad_dict else {})
        return const, new_lin, new_quad

    def _objective(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst

        const1, lin1 = self._linear_expression(src.objective.linear)
        const2, lin2, quadratic = self._quadratic_expression(src.objective.quadratic)

        constant = fsum([src.objective.constant] + const1 + const2)
        linear = lin1.coefficients + lin2.coefficients
        if src.objective.sense == src.objective.sense.minimize:
            dst.minimize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        else:
            dst.maximize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        return SubstitutionStatus.success

    def _linear_constraints(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst

        for lin_cst in src.linear_constraints:
            constant, linear = self._linear_expression(lin_cst.linear)
            rhs = lin_cst.rhs - fsum(constant)
            if linear.coefficients.nnz > 0:
                dst.linear_constraint(name=lin_cst.name, coefficients=linear.coefficients,
                                      sense=lin_cst.sense, rhs=rhs)
            else:
                if not self._feasible(lin_cst.sense, rhs):
                    logger.warning('constraint %s is infeasible due to substitution', lin_cst.name)
                    return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

    def _quadratic_constraints(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst

        for quad_cst in src.quadratic_constraints:
            const1, lin1 = self._linear_expression(quad_cst.linear)
            const2, lin2, quadratic = self._quadratic_expression(quad_cst.quadratic)
            rhs = quad_cst.rhs - fsum(const1 + const2)
            linear = lin1.coefficients + lin2.coefficients

            if quadratic.coefficients.nnz > 0:
                dst.quadratic_constraint(name=quad_cst.name, linear_coefficients=linear,
                                         quadratic_coefficients=quadratic.coefficients,
                                         sense=quad_cst.sense, rhs=rhs)
            elif linear.nnz > 0:
                name = quad_cst.name
                lin_names = set(lin.name for lin in dst.linear_constraints)
                while name in lin_names:
                    name = '_' + name
                dst.linear_constraint(name=name, coefficients=linear, sense=quad_cst.sense, rhs=rhs)
            else:
                if not self._feasible(quad_cst.sense, rhs):
                    logger.warning('constraint %s is infeasible due to substitution', quad_cst.name)
                    return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

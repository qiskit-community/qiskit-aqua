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

"""The inequality to equality converter."""

import copy
import math
from typing import List, Tuple, Dict, Optional

from cplex import SparsePair

from ..problems.quadratic_program import QuadraticProgram
from ..results.optimization_result import OptimizationResult
from ..utils.qiskit_optimization_error import QiskitOptimizationError


class InequalityToEqualityConverter:
    """Convert inequality constraints into equality constraints by introducing slack variables.

    Examples:
        >>> problem = QuadraticProgram()
        >>> # define a problem
        >>> conv = InequalityToEqualityConverter()
        >>> problem2 = conv.encode(problem)
    """

    _delimiter = '@'  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        """Initialize the integral variables."""

        self._src = None
        self._dst = None
        self._conv: Dict[str, List[Tuple[str, int]]] = {}
        # e.g., self._conv = {'c1': [c1@slack_var]}

    def encode(self, op: QuadraticProgram, name: Optional[str] = None,
               mode: str = 'auto') -> QuadraticProgram:
        """Convert a problem with inequality constraints into one with only equality constraints.

        Args:
            op: The problem to be solved, that may contain inequality constraints.
            name: The name of the converted problem.
            mode: To chose the type of slack variables. There are 3 options for mode.
                  - 'integer': All slack variables will be integer variables.
                  - 'continuous': All slack variables will be continuous variables
                  - 'auto': Try to use integer variables but if it's not possible,
                    use continuous variables

        Returns:
            The converted problem, that contain only equality constraints.

        Raises:
            QiskitOptimizationError: If a variable type is not supported.
            QiskitOptimizationError: If an unsupported mode is selected.
            QiskitOptimizationError: If an unsupported sense is specified.
        """
        self._src = copy.deepcopy(op)
        self._dst = QuadraticProgram()

        # declare variables
        names = self._src.variables.get_names()
        types = self._src.variables.get_types()
        lower_bounds = self._src.variables.get_lower_bounds()
        upper_bounds = self._src.variables.get_upper_bounds()

        for i, variable in enumerate(names):
            typ = types[i]
            if typ == 'B':
                self._dst.variables.add(names=[variable], types='B')
            elif typ in ['C', 'I']:
                self._dst.variables.add(names=[variable], types=typ,
                                        lb=[lower_bounds[i]],
                                        ub=[upper_bounds[i]])
            else:
                raise QiskitOptimizationError('Variable type not supported: ' + typ)

        # set objective name
        if name is None:
            self._dst.objective.set_name(self._src.objective.get_name())
        else:
            self._dst.objective.set_name(name)

        # set objective sense
        self._dst.objective.set_sense(self._src.objective.get_sense())

        # set objective offset
        self._dst.objective.set_offset(self._src.objective.get_offset())

        # set linear objective terms
        for i, v in self._src.objective.get_linear_dict().items():
            self._dst.objective.set_linear(i, v)

        # set quadratic objective terms
        for (i, j), v in self._src.objective.get_quadratic_dict().items():
            self._dst.objective.set_quadratic_coefficients(i, j, v)

        # set linear constraints
        names = self._src.linear_constraints.get_names()
        rows = self._src.linear_constraints.get_rows()
        senses = self._src.linear_constraints.get_senses()
        rhs = self._src.linear_constraints.get_rhs()

        for i, variable in enumerate(names):
            # Copy equality constraints into self._dst
            if senses[i] == 'E':
                self._dst.linear_constraints.add(lin_expr=[rows[i]], senses=[senses[i]],
                                                 rhs=[rhs[i]], names=[variable])
            # When the type of a constraint is L, make an equality constraint
            # with slack variables which represent [lb, ub] = [0, constant - the lower bound of lhs]
            elif senses[i] == 'L':
                if mode == 'integer':
                    self._add_int_slack_var_constraint(name=variable, row=rows[i], rhs=rhs[i],
                                                       sense=senses[i])
                elif mode == 'continuous':
                    self._add_continuous_slack_var_constraint(name=variable, row=rows[i],
                                                              rhs=rhs[i], sense=senses[i])
                elif mode == 'auto':
                    self._add_auto_slack_var_constraint(name=variable, row=rows[i], rhs=rhs[i],
                                                        sense=senses[i])
                else:
                    raise QiskitOptimizationError('Unsupported mode is selected' + mode)

            # When the type of a constraint is G, make an equality constraint
            # with slack variables which represent [lb, ub] = [0, the upper bound of lhs]
            elif senses[i] == 'G':
                if mode == 'integer':
                    self._add_int_slack_var_constraint(name=variable, row=rows[i], rhs=rhs[i],
                                                       sense=senses[i])
                elif mode == 'continuous':
                    self._add_continuous_slack_var_constraint(name=variable, row=rows[i],
                                                              rhs=rhs[i], sense=senses[i])
                elif mode == 'auto':
                    self._add_auto_slack_var_constraint(name=variable, row=rows[i], rhs=rhs[i],
                                                        sense=senses[i])
                else:
                    raise QiskitOptimizationError(
                        'Unsupported mode is selected' + mode)

            else:
                raise QiskitOptimizationError('Type of sense in ' + variable + 'is not supported')

        # TODO: add quadratic constraints
        if self._src.quadratic_constraints.get_num() > 0:
            raise QiskitOptimizationError('Quadratic constraints are not yet supported.')

        return self._dst

    def decode(self, result: OptimizationResult) -> OptimizationResult:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            result: The result of the converted problem.

        Returns:
            The result of the original problem.
        """
        new_result = OptimizationResult()
        # convert the optimization result into that of the original problem
        names = self._dst.variables.get_names()
        vals = result.x
        new_vals = self._decode_var(names, vals)
        new_result.x = new_vals
        new_result.fval = result.fval
        return new_result

    def _decode_var(self, names, vals) -> List[int]:
        # decode slack variables
        sol = {name: vals[i] for i, name in enumerate(names)}
        new_vals = []
        slack_var_names = []

        for lst in self._conv.values():
            slack_var_names.extend(lst)

        for name in sol:
            if name in slack_var_names:
                pass
            else:
                new_vals.append(sol[name])
        return new_vals

    def _add_int_slack_var_constraint(self, name, row, rhs, sense):
        # If a coefficient that is not integer exist, raise error
        if any(isinstance(coef, float) and not coef.is_integer() for coef in row.val):
            raise QiskitOptimizationError('Can not use a slack variable for ' + name)

        slack_name = name + self._delimiter + 'int_slack'
        lhs_lb, lhs_ub = self._calc_bounds(row)

        # If rhs is float number, round up/down to the nearest integer.
        if sense == 'L':
            new_rhs = math.floor(rhs)
        if sense == 'G':
            new_rhs = math.ceil(rhs)

        # Add a new integer variable.
        if sense == 'L':
            sign = 1
            self._dst.variables.add(names=[slack_name], lb=[0], ub=[new_rhs - lhs_lb], types=['I'])
        elif sense == 'G':
            sign = -1
            self._dst.variables.add(names=[slack_name], lb=[0], ub=[lhs_ub - new_rhs], types=['I'])
        else:
            raise QiskitOptimizationError('The type of Sense in ' + name + 'is not supported')

        self._conv[name] = slack_name

        new_ind = copy.deepcopy(row.ind)
        new_val = copy.deepcopy(row.val)

        new_ind.append(self._dst.variables.get_indices(slack_name))
        new_val.append(sign)

        # Add a new equality constraint.
        self._dst.linear_constraints.add(lin_expr=[SparsePair(ind=new_ind, val=new_val)],
                                         senses=['E'], rhs=[new_rhs], names=[name])

    def _add_continuous_slack_var_constraint(self, name, row, rhs, sense):
        slack_name = name + self._delimiter + 'continuous_slack'
        lhs_lb, lhs_ub = self._calc_bounds(row)

        if sense == 'L':
            sign = 1
            self._dst.variables.add(names=[slack_name], lb=[0], ub=[rhs - lhs_lb], types=['C'])
        elif sense == 'G':
            sign = -1
            self._dst.variables.add(names=[slack_name], lb=[0], ub=[lhs_ub - rhs], types=['C'])
        else:
            raise QiskitOptimizationError('Type of Sense in ' + name + 'is not supported')

        self._conv[name] = slack_name

        new_ind = copy.deepcopy(row.ind)
        new_val = copy.deepcopy(row.val)

        new_ind.append(self._dst.variables.get_indices(slack_name))
        new_val.append(sign)

        # Add a new equality constraint.
        self._dst.linear_constraints.add(lin_expr=[SparsePair(ind=new_ind, val=new_val)],
                                         senses=['E'], rhs=[rhs], names=[name])

    def _add_auto_slack_var_constraint(self, name, row, rhs, sense):
        # If a coefficient that is not integer exist, use a continuous slack variable
        if any(isinstance(coef, float) and not coef.is_integer() for coef in row.val):
            self._add_continuous_slack_var_constraint(name=name, row=row, rhs=rhs, sense=sense)
        # Else use an integer slack variable
        else:
            self._add_int_slack_var_constraint(name=name, row=row, rhs=rhs, sense=sense)

    def _calc_bounds(self, row):
        lhs_lb, lhs_ub = 0, 0
        for ind, val in zip(row.ind, row.val):
            if self._dst.variables.get_types(ind) == 'B':
                upper_bound = 1
            else:
                upper_bound = self._dst.variables.get_upper_bounds(ind)
            lower_bound = self._dst.variables.get_lower_bounds(ind)

            lhs_lb += min(lower_bound * val, upper_bound * val)
            lhs_ub += max(lower_bound * val, upper_bound * val)

        return lhs_lb, lhs_ub

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

"""The converter to convert an integer problem to a binary problem."""

import copy
from typing import List, Tuple, Dict, Optional

import numpy as np
from cplex import SparsePair

from ..problems.optimization_problem import OptimizationProblem
from ..results.optimization_result import OptimizationResult


class IntegerToBinaryConverter:
    """Convert an `OptimizationProblem` into new one by encoding integer with binary variables.

    Examples:
        >>> problem = OptimizationProblem()
        >>> problem.variables.add(names=['x'], types=['I'], lb=[0], ub=[10])
        >>> conv = IntegerToBinaryConverter()
        >>> problem2 = conv.encode(problem)
    """

    _delimiter = '@'  # users are supposed not to use this character in variable names

    def __init__(self) -> None:
        """Initializes the internal data structure."""
        self._src = None
        self._dst = None
        self._conv: Dict[str, List[Tuple[str, int]]] = {}
        # e.g., self._conv = {'x': [('x@1', 1), ('x@2', 2)]}

    def encode(self, op: OptimizationProblem, name: Optional[str] = None) -> OptimizationProblem:
        """Convert an integer problem into a new problem with binary variables.

        Args:
            op: The problem to be solved, that may contain integer variables.
            name: The name of the converted problem. If not provided, the name of the input
                problem is used.

        Returns:
            The converted problem, that contains no integer variables.
        """

        self._src = copy.deepcopy(op)
        self._dst = OptimizationProblem()
        if name:
            self._dst.set_problem_name(name)
        else:
            self._dst.set_problem_name(self._src.get_problem_name())

        # declare variables
        names = self._src.variables.get_names()
        types = self._src.variables.get_types()
        lower_bounds = self._src.variables.get_lower_bounds()
        upper_bounds = self._src.variables.get_upper_bounds()
        for i, variable in enumerate(names):
            typ = types[i]
            if typ == 'I':
                new_vars: List[Tuple[str, int]] = self._encode_var(name=variable,
                                                                   lower_bound=lower_bounds[i],
                                                                   upper_bound=upper_bounds[i])
                self._conv[variable] = new_vars
                self._dst.variables.add(names=[new_name for new_name, _ in new_vars],
                                        types='B' * len(new_vars))
            else:
                self._dst.variables.add(names=[variable], types=typ,
                                        lb=[lower_bounds[i]], ub=[upper_bounds[i]])

        # replace integer variables with binary variables in the objective function
        # self.objective.subs(self._conv)

        # replace integer variables with binary variables in the constrains
        # self.linear_constraints.subs(self._conv)
        # self.quadratic_constraints.subs(self._conv)
        # note: `subs` substitutes variables with sets of auxiliary variables

        self._substitute_int_var()

        return self._dst

    def _encode_var(self, name: str, lower_bound: int, upper_bound: int) -> List[Tuple[str, int]]:
        # bounded-coefficient encoding proposed in arxiv:1706.01945 (Eq. (5))
        var_range = upper_bound - lower_bound
        power = int(np.log2(var_range))
        bounded_coef = var_range - (2 ** power - 1)

        lst = []
        for i in range(power):
            coef = 2 ** i
            new_name = name + self._delimiter + str(i)
            lst.append((new_name, coef))

        new_name = name + self._delimiter + str(power)
        lst.append((new_name, bounded_coef))

        return lst

    def _substitute_int_var(self):
        # set objective name
        self._dst.objective.set_name(self._src.objective.get_name())

        # set the sense of the objective function
        self._dst.objective.set_sense(self._src.objective.get_sense())

        # set offset
        self._dst.objective.set_offset(self._src.objective.get_offset())

        # set linear terms of objective function
        src_obj_linear = self._src.objective.get_linear()

        for src_var_index in src_obj_linear:
            coef = src_obj_linear[src_var_index]
            var_name = self._src.variables.get_names(src_var_index)

            if var_name in self._conv:
                for converted_name, converted_coef in self._conv[var_name]:
                    self._dst.objective.set_linear(converted_name, coef * converted_coef)

            else:
                self._dst.objective.set_linear(var_name, coef)

        # set quadratic terms of objective function
        src_obj_quad = self._src.objective.get_quadratic()

        num_var = self._dst.variables.get_num()
        new_quad = np.zeros((num_var, num_var))

        for row in src_obj_quad:
            for col in src_obj_quad[row]:
                row_var_name = self._src.variables.get_names(row)
                col_var_name = self._src.variables.get_names(col)
                coef = src_obj_quad[row][col]

                if row_var_name in self._conv:
                    row_vars = self._conv[row_var_name]
                else:
                    row_vars = [(row_var_name, 1)]

                if col_var_name in self._conv:
                    col_vars = self._conv[col_var_name]
                else:
                    col_vars = [(col_var_name, 1)]

                for new_row, row_coef in row_vars:
                    for new_col, col_coef in col_vars:
                        row_index = self._dst.variables.get_indices(new_row)
                        col_index = self._dst.variables.get_indices(new_col)
                        new_quad[row_index, col_index] = coef * row_coef * col_coef

        ind = list(range(num_var))
        lst = []
        for i in ind:
            lst.append(SparsePair(ind=ind, val=new_quad[i].tolist()))
        self._dst.objective.set_quadratic(lst)

        # set constraints whose integer variables are replaced with binary variables
        linear_rows = self._src.linear_constraints.get_rows()
        linear_sense = self._src.linear_constraints.get_senses()
        linear_rhs = self._src.linear_constraints.get_rhs()
        linear_ranges = self._src.linear_constraints.get_range_values()
        linear_names = self._src.linear_constraints.get_names()

        lin_expr = []

        for i, linear_row in enumerate(linear_rows):
            sparse_pair = SparsePair()
            for j, var_ind in enumerate(linear_row.ind):
                coef = linear_row.val[j]
                var_name = self._src.variables.get_names(var_ind)

                if var_name in self._conv:
                    for converted_name, converted_coef in self._conv[var_name]:
                        sparse_pair.ind.append(converted_name)
                        sparse_pair.val.append(converted_coef * coef)
                else:
                    sparse_pair.ind.append(var_name)
                    sparse_pair.val.append(coef)

            lin_expr.append(sparse_pair)

        self._dst.linear_constraints.add(lin_expr, linear_sense, linear_rhs, linear_ranges,
                                         linear_names)

    def decode(self, result: OptimizationResult) -> OptimizationResult:
        """Convert the encoded problem (binary variables) back to the original (integer variables).

        Args:
            result: The result of the converted problem.

        Returns:
            The result of the original problem.
        """
        names = self._dst.variables.get_names()
        vals = result.x
        new_vals = self._decode_var(names, vals)
        result.x = new_vals
        return result

    def _decode_var(self, names, vals) -> List[int]:
        # decode integer values
        sol = {name: int(vals[i]) for i, name in enumerate(names)}
        new_vals = []
        for name in self._src.variables.get_names():
            if name in self._conv:
                new_vals.append(sum(sol[aux] * coef for aux, coef in self._conv[name]))
            else:
                new_vals.append(sol[name])
        return new_vals

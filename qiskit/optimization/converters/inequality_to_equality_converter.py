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


import copy
from typing import List, Tuple, Dict

import numpy as np

from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from qiskit.optimization.results.optimization_result import OptimizationResult
from qiskit.optimization.utils import QiskitOptimizationError


class InequalityToEqualityConverter:
    """ Convert inequality constraints into equality constraints by introducing slack variables.

        Examples:
            >>> problem = OptimizationProblem()
            >>> # define a problem
            >>> conv = InequalityToEqualityConverter()
            >>> problem2 = conv.encode(problem)
    """

    _delimiter = '@'  # users are supposed not to use this character in variable names

    def __init__(self):
        """ Constructor. It initializes the internal data structure. No args.
        """
        self._src = None
        self._dst = None
        self._conv: Dict[str, List[Tuple[str, int]]] = {}
        # e.g., self._conv = {'c1': [('s@1', 1), ('s@2', 2)]}

    def encode(self, op: OptimizationProblem, name: str = None) -> OptimizationProblem:
        """ Convert a problem with inequality constraints into new one with only equality
        constraints.

        Args:
            op: The problem to be solved, that may contain inequality constraints.
            name: The name of the converted problem.

        Returns:
            The converted problem, that contain only equality constraints.

        """
        self._src = copy.deepcopy(op)
        self._dst = OptimizationProblem()

        # declare variables
        names = self._src.variables.get_names()
        types = self._src.variables.get_types()
        lb = self._src.variables.get_lower_bounds()
        ub = self._src.variables.get_upper_bounds()

        for i, name in enumerate(names):
            typ = types[i]
            if typ == 'B':
                self._dst.variables.add(names=[name], types='B')
            elif typ == 'C' or typ == 'I':
                self._dst.variables.add(names=[name], types=typ, lb=[lb[i]], ub=[ub[i]])
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
        for i, v in self._src.objective.get_linear().items():
            self._dst.objective.set_linear(i, v)

        # set quadratic objective terms
        for i, vi in self._src.objective.get_quadratic().items():
            for j, v in vi.items():
                self._dst.objective.set_quadratic_coefficients(i, j, v)

        # set linear constraints
        names = self._src.linear_constraints.get_names()
        rows = self._src.linear_constraints.get_rows()
        senses = self._src.linear_constraints.get_senses()
        rhs = self._src.linear_constraints.get_rhs()

        for i, name in enumerate(names):
            # Copy equality constraints into self._dst
            if senses[i] == 'E':
                self._dst.linear_constraints.add(lin_expr=[rows[i]], senses=[senses[i]],
                                                 rhs=[rhs[i]], names=[names[i]])
            # When the type of a constraint is L, make an equality constraint
            # with slack variables which represent [lb, ub] = [0, constant - the lower bound of lhs]
            elif senses[i] == 'L':
                lhs_lb = 0
                for ind, val in zip(rows[i].ind, rows[i].val):
                    if self._dst.variables.get_types(ind) == 'B':
                        ub = 1
                    else:
                        ub = self._dst.variables.get_upper_bounds(ind)
                    lb = self._dst.variables.get_lower_bounds(ind)

                    lhs_lb += min(lb * val, ub * val)

                slack_vars = self._encode_var(name=name + '_slack', lb=0, ub=rhs[i] - lhs_lb)
                self._dst.variables.add(names=[name for name, _ in slack_vars],
                                        types='B' * len(slack_vars))
                self._conv[names[i]] = slack_vars

                new_ind = rows[i].ind
                new_val = rows[i].val

                for name, coef in slack_vars:
                    new_ind.append(self._dst.variables._varsgetindex[name])
                    new_val.append(coef)
                self._dst.linear_constraints.add(lin_expr=[rows[i]], senses=['E'], rhs=[rhs[i]],
                                                 names=[names[i]])

            # When the type of a constraint is G, make an equality constraint
            # with slack variables which represent [lb, ub] = [0, the upper bound of lhs]
            elif senses[i] == 'G':
                lhs_ub = 0
                for ind, val in zip(rows[i].ind, rows[i].val):
                    if self._dst.variables.get_types(ind) == 'B':
                        ub = 1
                    else:
                        ub = self._dst.variables.get_upper_bounds(ind)
                    lb = self._dst.variables.get_lower_bounds(ind)

                    lhs_ub += max(lb * val, ub * val)
                slack_vars = self._encode_var(name=name + '_slack', lb=0, ub=lhs_ub - rhs[i])
                self._dst.variables.add(names=[name for name, _ in slack_vars],
                                        types='B' * len(slack_vars))
                self._conv[names[i]] = slack_vars

                new_ind = rows[i].ind
                new_val = rows[i].val

                for name, coef in slack_vars:
                    new_ind.append(self._dst.variables._varsgetindex[name])
                    new_val.append(-1 * coef)
                self._dst.linear_constraints.add(lin_expr=[rows[i]], senses=['E'], rhs=[rhs[i]],
                                                 names=[names[i]])

            else:
                raise QiskitOptimizationError('Sense type not supported: ' + senses[i])

        return self._dst

    def _encode_var(self, name: str, lb: int, ub: int) -> List[Tuple[str, int]]:
        # bounded-coefficient encoding proposed in arxiv:1706.01945 (Eq. (5))
        var_range = ub - lb
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

    def decode(self, result: OptimizationResult) -> OptimizationResult:
        """ Convert a result of a converted problem into that of the original problem.

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

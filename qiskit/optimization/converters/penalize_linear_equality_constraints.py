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

"""Converter to convert a problem with equality constraints to unconstrained with penalty terms."""

from typing import Optional

import copy
from collections import defaultdict

from ..problems.optimization_problem import OptimizationProblem
from ..utils import QiskitOptimizationError


class PenalizeLinearEqualityConstraints:
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self):
        """Initialize the internal data structure."""
        self._src = None
        self._dst = None

    def encode(self, op: OptimizationProblem, penalty_factor: float = 1e5,
               name: Optional[str] = None) -> OptimizationProblem:
        """Convert a problem with equality constraints into an unconstrained problem.

        Args:
            op: The problem to be solved, that does not contain inequality constraints.
            penalty_factor: Penalty terms in the objective function is multiplied with this factor.
            name: The name of the converted problem.

        Returns:
            The converted problem, that is an unconstrained problem.

        Raises:
            QiskitOptimizationError: If an inequality constraint exists.
        """

        # TODO: test compatibility, how to react in case of incompatibility?

        # create empty OptimizationProblem model
        self._src = copy.deepcopy(op)  # deep copy
        self._dst = OptimizationProblem()

        # set variables (obj is set via objective interface)
        var_names = self._src.variables.get_names()
        var_lbs = self._src.variables.get_lower_bounds()
        var_ubs = self._src.variables.get_upper_bounds()
        var_types = self._src.variables.get_types()
        if var_names:
            self._dst.variables.add(lb=var_lbs, ub=var_ubs, types=var_types, names=var_names)

        # set objective name
        if name is None:
            self._dst.set_problem_name(self._src.get_problem_name())
        else:
            self._dst.set_problem_name(name)

        # set objective sense
        self._dst.objective.set_sense(self._src.objective.get_sense())
        penalty_factor = self._src.objective.get_sense() * penalty_factor

        # store original objective offset
        offset = self._src.objective.get_offset()

        # store original linear objective terms
        linear_terms = defaultdict(int)
        for i, v in self._src.objective.get_linear().items():
            linear_terms[i] = v

        # store original quadratic objective terms
        quadratic_terms = defaultdict(lambda: defaultdict(int))
        for i, v in self._src.objective.get_quadratic().items():
            quadratic_terms[i].update(v)

        # get linear constraints' data
        linear_rows = self._src.linear_constraints.get_rows()
        linear_sense = self._src.linear_constraints.get_senses()
        linear_rhs = self._src.linear_constraints.get_rhs()
        linear_names = self._src.linear_constraints.get_names()

        # if inequality constraints exist, raise an error
        if not all(ls == 'E' for ls in linear_sense):
            raise QiskitOptimizationError('An inequality constraint exists. '
                                          'The method supports only equality constraints.')

        # convert linear constraints into penalty terms
        num_constraints = len(linear_names)
        for i in range(num_constraints):
            constant = linear_rhs[i]
            row = linear_rows[i]

            # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
            offset += penalty_factor * constant ** 2

            # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
            for var_ind, coef in zip(row.ind, row.val):
                # if var_ind already exisits in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict
                linear_terms[var_ind] += penalty_factor * -2 * coef * constant

            # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
            for var_ind_1, coef_1 in zip(row.ind, row.val):
                for var_ind_2, coef_2 in zip(row.ind, row.val):
                    # if var_ind_1 and var_ind_2 already exisit in the quadratic terms dic,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    # according to implementation of quadratic terms in OptimizationModel,
                    # multiply by 2
                    quadratic_terms[var_ind_1][var_ind_2] += penalty_factor * coef_1 * coef_2 * 2

        # set objective offset
        self._dst.objective.set_offset(offset)

        # set linear objective terms
        for i, v in linear_terms.items():
            self._dst.objective.set_linear(i, v)

        # set quadratic objective terms
        for i, v_i in quadratic_terms.items():
            for j, v in v_i.items():
                self._dst.objective.set_quadratic_coefficients(i, j, v)

        return self._dst

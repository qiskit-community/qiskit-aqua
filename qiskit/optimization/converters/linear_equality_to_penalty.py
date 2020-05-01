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

import copy
from typing import Optional

from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable
from ..problems.constraint import Constraint
from ..problems.quadratic_objective import QuadraticObjective
from ..exceptions import QiskitOptimizationError


class LinearEqualityToPenalty:
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self):
        self._src = None
        self._dst = None

    def encode(self, op: QuadraticProgram, penalty_factor: float = 1e5,
               name: Optional[str] = None) -> QuadraticProgram:
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

        # create empty QuadraticProgram model
        self._src = copy.deepcopy(op)  # deep copy
        self._dst = QuadraticProgram()

        # set variables
        for x in self._src.variables:
            if x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(x.lowerbound, x.upperbound, x.name)
            elif x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(x.lowerbound, x.upperbound, x.name)
            else:
                raise QiskitOptimizationError('Unsupported vartype: {}'.format(x.vartype))

        # set problem name
        if name is None:
            self._dst.name = self._src.name
        else:
            self._dst.name = name

        # get original objective terms
        offset = self._src.objective.constant
        linear = self._src.objective.linear.to_dict()
        quadratic = self._src.objective.quadratic.to_dict()
        sense = self._src.objective.sense.value

        # convert linear constraints into penalty terms
        for constraint in self._src.linear_constraints:

            if constraint.sense != Constraint.Sense.EQ:
                raise QiskitOptimizationError('An inequality constraint exists. '
                                              'The method supports only equality constraints.')

            constant = constraint.rhs
            row = constraint.linear.to_dict()

            # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
            offset += sense * penalty_factor * constant**2

            # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
            for j, coef in row.items():
                # if j already exists in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict
                linear[j] = linear.get(j, 0.0) + sense * penalty_factor * -2 * coef * constant

            # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
            for j, coef_1 in row.items():
                for k, coef_2 in row.items():
                    # if j and k already exist in the quadratic terms dict,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    # according to implementation of quadratic terms in OptimizationModel,
                    # don't need to multiply by 2, since loops run over (x, y) and (y, x).
                    quadratic[(j, k)] = quadratic.get((j, k), 0.0) \
                        + sense * penalty_factor * coef_1 * coef_2

        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(offset, linear, quadratic)
        else:
            self._dst.maximize(offset, linear, quadratic)

        return self._dst

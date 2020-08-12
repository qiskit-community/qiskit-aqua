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
import logging
from math import fsum
from typing import Optional, cast, Union, Tuple, Dict

from ..algorithms.optimization_algorithm import OptimizationResult, OptimizationResultStatus
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram, QuadraticProgramStatus
from ..problems.variable import Variable
from .quadratic_program_converter import QuadraticProgramConverter

logger = logging.getLogger(__name__)


class LinearEqualityToPenalty(QuadraticProgramConverter):
    """Convert a problem with only equality constraints to unconstrained with penalty terms."""

    def __init__(self, penalty: Optional[float] = None) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                     If None is passed, penalty factor will be automatically calculated.
        """
        self._src = None  # type: Optional[QuadraticProgram]
        self._dst = None  # type: Optional[QuadraticProgram]
        self._penalty = penalty  # type: Optional[float]

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with equality constraints into an unconstrained problem.

        Args:
            problem: The problem to be solved, that does not contain inequality constraints.

        Returns:
            The converted problem, that is an unconstrained problem.

        Raises:
            QiskitOptimizationError: If an inequality constraint exists.
        """

        # create empty QuadraticProgram model
        self._src = copy.deepcopy(problem)
        self._dst = QuadraticProgram(name=problem.name)

        # If penalty is None, set the penalty coefficient by _auto_define_penalty()
        if self._penalty is None:
            penalty = self._auto_define_penalty()
        else:
            penalty = self._penalty

        # Set variables
        for x in self._src.variables:
            if x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(x.lowerbound, x.upperbound, x.name)
            elif x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(x.lowerbound, x.upperbound, x.name)
            else:
                raise QiskitOptimizationError('Unsupported vartype: {}'.format(x.vartype))

        # get original objective terms
        offset = self._src.objective.constant
        linear = self._src.objective.linear.to_dict()
        quadratic = self._src.objective.quadratic.to_dict()
        sense = self._src.objective.sense.value

        # convert linear constraints into penalty terms
        for constraint in self._src.linear_constraints:

            if constraint.sense != Constraint.Sense.EQ:
                raise QiskitOptimizationError(
                    'An inequality constraint exists. '
                    'The method supports only equality constraints.'
                )

            constant = constraint.rhs
            row = constraint.linear.to_dict()

            # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
            offset += sense * penalty * constant ** 2

            # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
            for j, coef in row.items():
                # if j already exists in the linear terms dic, add a penalty term
                # into existing value else create new key and value in the linear_term dict
                linear[j] = linear.get(j, 0.0) + sense * penalty * -2 * coef * constant

            # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
            for j, coef_1 in row.items():
                for k, coef_2 in row.items():
                    # if j and k already exist in the quadratic terms dict,
                    # add a penalty term into existing value
                    # else create new key and value in the quadratic term dict

                    # according to implementation of quadratic terms in OptimizationModel,
                    # don't need to multiply by 2, since loops run over (x, y) and (y, x).
                    tup = cast(Union[Tuple[int, int], Tuple[str, str]], (j, k))
                    quadratic[tup] = quadratic.get(tup, 0.0) + sense * penalty * coef_1 * coef_2

        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(offset, linear, quadratic)
        else:
            self._dst.maximize(offset, linear, quadratic)

        return self._dst

    def _auto_define_penalty(self) -> float:
        """Automatically define the penalty coefficient.

        Returns:
            Return the minimum valid penalty factor calculated
            from the upper bound and the lower bound of the objective function.
            If a constraint has a float coefficient,
            return the default value for the penalty factor.
        """
        default_penalty = 1e5

        # Check coefficients of constraints.
        # If a constraint has a float coefficient, return the default value for the penalty factor.
        terms = []
        for constraint in self._src.linear_constraints:
            terms.append(constraint.rhs)
            terms.extend(coef for coef in constraint.linear.to_dict().values())
        if any(isinstance(term, float) and not term.is_integer() for term in terms):
            logger.warning(
                'Warning: Using %f for the penalty coefficient because '
                'a float coefficient exists in constraints. \n'
                'The value could be too small. '
                'If so, set the penalty coefficient manually.',
                default_penalty,
            )
            return default_penalty

        # (upper bound - lower bound) can be calculate as the sum of absolute value of coefficients
        # Firstly, add 1 to guarantee that infeasible answers will be greater than upper bound.
        penalties = [1.0]
        # add linear terms of the object function.
        penalties.extend(abs(coef) for coef in self._src.objective.linear.to_dict().values())
        # add quadratic terms of the object function.
        penalties.extend(abs(coef) for coef in self._src.objective.quadratic.to_dict().values())

        return fsum(penalties)

    def interpret(self, result: OptimizationResult) -> OptimizationResult:
        """Convert the result of the converted problem back to that of the original problem

        Args:
            result: The result of the converted problem.

        Returns:
            The result of the original problem.

        Raises:
            QiskitOptimizationError: if the number of variables in the result differs from
                                     that of the original problem.
        """
        if len(result.x) != self._src.get_num_vars():
            raise QiskitOptimizationError(
                'The number of variables in the passed result differs from '
                'that of the original problem.'
            )
        # Substitute variables to obtain the function value and feasibility in the original problem
        substitute_dict = {}  # type: Dict[Union[str, int], float]
        variables = self._src.variables
        for i in range(len(result.x)):
            substitute_dict[variables[i].name] = float(result.x[i])
        substituted_qp = self._src.substitute_variables(substitute_dict)

        # Set the new status of optimization result
        if substituted_qp.status == QuadraticProgramStatus.VALID:
            new_status = OptimizationResultStatus.SUCCESS
        else:
            new_status = OptimizationResultStatus.INFEASIBLE

        return OptimizationResult(x=result.x, fval=substituted_qp.objective.constant,
                                  variables=self._src.variables, raw_results=result.raw_results,
                                  status=new_status)

    @property
    def penalty(self) -> Optional[float]:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: Optional[float]) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, penalty factor will be automatically calculated.
        """
        self._penalty = penalty

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

"""A converter from quadratic program to a QUBO."""

from typing import Optional, Union, List

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram


class QuadraticProgramToQubo(QuadraticProgramConverter):
    """Convert a given optimization problem to a new problem that is a QUBO.

        Examples:
            >>> from qiskit.optimization.problems import QuadraticProgram
            >>> from qiskit.optimization.converters import QuadraticProgramToQubo
            >>> problem = QuadraticProgram()
            >>> # define a problem
            >>> conv = QuadraticProgramToQubo()
            >>> problem2 = conv.convert(problem)
    """

    def __init__(self, penalty: Optional[float] = None) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
                If None is passed, penalty factor will be automatically calculated.
        """
        from ..converters.integer_to_binary import IntegerToBinary
        from ..converters.inequality_to_equality import InequalityToEquality
        from ..converters.linear_equality_to_penalty import LinearEqualityToPenalty

        self._int_to_bin = IntegerToBinary()
        self._ineq_to_eq = InequalityToEquality(mode='integer')
        self._penalize_lin_eq_constraints = LinearEqualityToPenalty(penalty=penalty)

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with linear equality constraints into new one with a QUBO form.

        Args:
            problem: The problem with linear equality constraints to be solved.

        Returns:
            The problem converted in QUBO format.

        Raises:
            QiskitOptimizationError: In case of an incompatible problem.
        """

        # analyze compatibility of problem
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))

        # Convert inequality constraints into equality constraints by adding slack variables
        problem_ = self._ineq_to_eq.convert(problem)

        # Map integer variables to binary variables
        problem_ = self._int_to_bin.convert(problem_)

        # Penalize linear equality constraints with only binary variables
        problem_ = self._penalize_lin_eq_constraints.convert(problem_)

        # Return QUBO
        return problem_

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

            Args:
                x: The result of the converted problem.

            Returns:
                The result of the original problem.
        """
        x = self._penalize_lin_eq_constraints.interpret(x)
        x = self._int_to_bin.interpret(x)
        x = self._ineq_to_eq.interpret(x)
        return x

    @staticmethod
    def get_compatibility_msg(problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """

        # initialize message
        msg = ''
        # check whether there are incompatible variable types
        if problem.get_num_continuous_vars() > 0:
            msg += 'Continuous variables are not supported! '

        # check whether there are incompatible constraint types
        if len(problem.quadratic_constraints) > 0:
            msg += 'Quadratic constraints are not supported. '
        # check whether there are float coefficients in constraints
        compatible_with_integer_slack = True
        for l_constraint in problem.linear_constraints:
            linear = l_constraint.linear.to_dict()
            if any(isinstance(coef, float) and not coef.is_integer() for coef in linear.values()):
                compatible_with_integer_slack = False
        for q_constraint in problem.quadratic_constraints:
            linear = q_constraint.linear.to_dict()
            quadratic = q_constraint.quadratic.to_dict()
            if any(
                    isinstance(coef, float) and not coef.is_integer()
                    for coef in quadratic.values()
            ) or any(
                isinstance(coef, float) and not coef.is_integer() for coef in linear.values()
            ):
                compatible_with_integer_slack = False
        if not compatible_with_integer_slack:
            msg += 'Can not convert inequality constraints to equality constraint because \
                    float coefficients are in constraints. '

        # if an error occurred, return error message, otherwise, return None
        return msg

    def is_compatible(self, problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0

    @property
    def penalty(self) -> Optional[float]:
        """Returns the penalty factor used in conversion.

        Returns:
            The penalty factor used in conversion.
        """
        return self._penalize_lin_eq_constraints.penalty

    @penalty.setter
    def penalty(self, penalty: Optional[float]) -> None:
        """Set a new penalty factor.

        Args:
            penalty: The new penalty factor.
                     If None is passed, penalty factor will be automatically calculated.
        """
        self._penalize_lin_eq_constraints.penalty = penalty

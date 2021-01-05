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
import logging
import math
from typing import List, Optional, Union

import numpy as np

from .quadratic_program_converter import QuadraticProgramConverter
from ..exceptions import QiskitOptimizationError
from ..problems.constraint import Constraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import Variable

logger = logging.getLogger(__name__)


class InequalityToEquality(QuadraticProgramConverter):
    """Convert inequality constraints into equality constraints by introducing slack variables.

    Examples:
        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> from qiskit.optimization.converters import InequalityToEquality
        >>> problem = QuadraticProgram()
        >>> # define a problem
        >>> conv = InequalityToEquality()
        >>> problem2 = conv.convert(problem)
    """

    _delimiter = '@'  # users are supposed not to use this character in variable names

    def __init__(self, mode: str = 'auto') -> None:
        """
        Args:
            mode: To chose the type of slack variables. There are 3 options for mode.

                - 'integer': All slack variables will be integer variables.
                - 'continuous': All slack variables will be continuous variables
                - 'auto': Try to use integer variables but if it's not possible,
                   use continuous variables
        """
        self._src = None  # type: Optional[QuadraticProgram]
        self._dst = None  # type: Optional[QuadraticProgram]
        self._mode = mode

    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with inequality constraints into one with only equality constraints.

        Args:
            problem: The problem to be solved, that may contain inequality constraints.

        Returns:
            The converted problem, that contain only equality constraints.

        Raises:
            QiskitOptimizationError: If a variable type is not supported.
            QiskitOptimizationError: If an unsupported mode is selected.
            QiskitOptimizationError: If an unsupported sense is specified.
        """
        self._src = copy.deepcopy(problem)
        self._dst = QuadraticProgram(name=problem.name)

        # set a converting mode
        mode = self._mode

        # Copy variables
        for x in self._src.variables:
            if x.vartype == Variable.Type.BINARY:
                self._dst.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                self._dst.integer_var(
                    name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound
                )
            elif x.vartype == Variable.Type.CONTINUOUS:
                self._dst.continuous_var(
                    name=x.name, lowerbound=x.lowerbound, upperbound=x.upperbound
                )
            else:
                raise QiskitOptimizationError(
                    "Unsupported variable type {}".format(x.vartype))

        # Copy the objective function
        constant = self._src.objective.constant
        linear = self._src.objective.linear.to_dict(use_name=True)
        quadratic = self._src.objective.quadratic.to_dict(use_name=True)
        if self._src.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            self._dst.minimize(constant, linear, quadratic)
        else:
            self._dst.maximize(constant, linear, quadratic)

        # For linear constraints
        for l_constraint in self._src.linear_constraints:
            linear = l_constraint.linear.to_dict(use_name=True)
            if l_constraint.sense == Constraint.Sense.EQ:
                self._dst.linear_constraint(
                    linear, l_constraint.sense, l_constraint.rhs, l_constraint.name
                )
            elif (
                    l_constraint.sense == Constraint.Sense.LE
                    or l_constraint.sense == Constraint.Sense.GE
            ):
                if mode == 'integer':
                    self._add_integer_slack_var_linear_constraint(
                        linear, l_constraint.sense, l_constraint.rhs, l_constraint.name
                    )
                elif mode == 'continuous':
                    self._add_continuous_slack_var_linear_constraint(
                        linear, l_constraint.sense, l_constraint.rhs, l_constraint.name
                    )
                elif mode == 'auto':
                    self._add_auto_slack_var_linear_constraint(
                        linear, l_constraint.sense, l_constraint.rhs, l_constraint.name
                    )
                else:
                    raise QiskitOptimizationError(
                        'Unsupported mode is selected: {}'.format(mode))
            else:
                raise QiskitOptimizationError(
                    'Type of sense in {} is not supported'.format(
                        l_constraint.name)
                )

        # For quadratic constraints
        for q_constraint in self._src.quadratic_constraints:
            linear = q_constraint.linear.to_dict(use_name=True)
            quadratic = q_constraint.quadratic.to_dict(use_name=True)
            if q_constraint.sense == Constraint.Sense.EQ:
                self._dst.quadratic_constraint(
                    linear, quadratic, q_constraint.sense, q_constraint.rhs, q_constraint.name
                )
            elif (
                    q_constraint.sense == Constraint.Sense.LE
                    or q_constraint.sense == Constraint.Sense.GE
            ):
                if mode == 'integer':
                    self._add_integer_slack_var_quadratic_constraint(
                        linear, quadratic, q_constraint.sense, q_constraint.rhs, q_constraint.name,
                    )
                elif mode == 'continuous':
                    self._add_continuous_slack_var_quadratic_constraint(
                        linear, quadratic, q_constraint.sense, q_constraint.rhs, q_constraint.name,
                    )
                elif mode == 'auto':
                    self._add_auto_slack_var_quadratic_constraint(
                        linear, quadratic, q_constraint.sense, q_constraint.rhs, q_constraint.name,
                    )
                else:
                    raise QiskitOptimizationError(
                        'Unsupported mode is selected: {}'.format(mode))
            else:
                raise QiskitOptimizationError(
                    'Type of sense in {} is not supported'.format(
                        q_constraint.name)
                )

        return self._dst

    def _add_integer_slack_var_linear_constraint(self, linear, sense, rhs, name):
        # If a coefficient that is not integer exist, raise error
        if self._contains_any_float_value(linear.values()):
            raise QiskitOptimizationError(
                '"{0}" contains float coefficients. '
                'We can not use an integer slack variable for "{0}"'.format(name))

        # If rhs is float number, round up/down to the nearest integer.
        new_rhs = rhs
        if sense == Constraint.Sense.LE:
            new_rhs = math.floor(rhs)
        if sense == Constraint.Sense.GE:
            new_rhs = math.ceil(rhs)

        # Add a new integer variable.
        slack_name = name + self._delimiter + 'int_slack'

        lhs_lb, lhs_ub = self._calc_linear_bounds(linear)

        var_added = False

        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
                self._dst.integer_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1
                self._dst.integer_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        else:
            raise QiskitOptimizationError(
                'The type of Sense in {} is not supported'.format(name))

        # Add a new equality constraint.
        new_linear = copy.deepcopy(linear)
        if var_added:
            new_linear[slack_name] = sign
        self._dst.linear_constraint(new_linear, "==", new_rhs, name)

    def _add_continuous_slack_var_linear_constraint(self, linear, sense, rhs, name):
        slack_name = name + self._delimiter + 'continuous_slack'

        lhs_lb, lhs_ub = self._calc_linear_bounds(linear)

        var_added = False
        if sense == Constraint.Sense.LE:
            var_ub = rhs - lhs_lb
            if var_ub > 0:
                sign = 1
                self._dst.continuous_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - rhs
            if var_ub > 0:
                sign = -1
                self._dst.continuous_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        else:
            raise QiskitOptimizationError(
                'The type of Sense in {} is not supported'.format(name))

        # Add a new equality constraint.
        new_linear = copy.deepcopy(linear)
        if var_added:
            new_linear[slack_name] = sign
        self._dst.linear_constraint(new_linear, "==", rhs, name)

    def _add_auto_slack_var_linear_constraint(self, linear, sense, rhs, name):
        # If a coefficient that is not integer exist, use a continuous slack variable
        if self._contains_any_float_value(list(linear.values())):
            self._add_continuous_slack_var_linear_constraint(
                linear, sense, rhs, name)
        # Else use an integer slack variable
        else:
            self._add_integer_slack_var_linear_constraint(
                linear, sense, rhs, name)

    def _add_integer_slack_var_quadratic_constraint(self, linear, quadratic, sense, rhs, name):
        # If a coefficient that is not integer exist, raise an error
        if (self._contains_any_float_value(list(linear.values()))
                or self._contains_any_float_value(list(quadratic.values()))):
            raise QiskitOptimizationError(
                '"{0}" contains float coefficients. '
                'We can not use an integer slack variable for "{0}"'.format(name))

        # If rhs is float number, round up/down to the nearest integer.
        new_rhs = rhs
        if sense == Constraint.Sense.LE:
            new_rhs = math.floor(rhs)
        if sense == Constraint.Sense.GE:
            new_rhs = math.ceil(rhs)

        # Add a new integer variable.
        slack_name = name + self._delimiter + 'int_slack'

        lhs_lb, lhs_ub = self._calc_quadratic_bounds(linear, quadratic)

        var_added = False

        if sense == Constraint.Sense.LE:
            var_ub = new_rhs - lhs_lb
            if var_ub > 0:
                sign = 1
                self._dst.integer_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - new_rhs
            if var_ub > 0:
                sign = -1
                self._dst.integer_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        else:
            raise QiskitOptimizationError(
                'The type of Sense in {} is not supported'.format(name))

        # Add a new equality constraint.
        new_linear = copy.deepcopy(linear)
        if var_added:
            new_linear[slack_name] = sign
        self._dst.quadratic_constraint(
            new_linear, quadratic, "==", new_rhs, name)

    def _add_continuous_slack_var_quadratic_constraint(self, linear, quadratic, sense, rhs, name):
        # Add a new continuous variable.
        slack_name = name + self._delimiter + 'continuous_slack'

        lhs_lb, lhs_ub = self._calc_quadratic_bounds(linear, quadratic)

        var_added = False

        if sense == Constraint.Sense.LE:
            var_ub = rhs - lhs_lb
            if var_ub > 0:
                sign = 1
                self._dst.continuous_var(
                    name=slack_name, lowerbound=0, upperbound=var_ub)
                var_added = True
        elif sense == Constraint.Sense.GE:
            var_ub = lhs_ub - rhs
            if var_ub > 0:
                sign = -1
                self._dst.continuous_var(
                    name=slack_name, lowerbound=0, upperbound=lhs_ub - rhs)
                var_added = True
        else:
            raise QiskitOptimizationError(
                'The type of Sense in {} is not supported'.format(name))

        # Add a new equality constraint.
        new_linear = copy.deepcopy(linear)
        if var_added:
            new_linear[slack_name] = sign
        self._dst.quadratic_constraint(new_linear, quadratic, "==", rhs, name)

    def _add_auto_slack_var_quadratic_constraint(self, linear, quadratic, sense, rhs, name):
        # If a coefficient that is not integer exist, use a continuous slack variable
        if (self._contains_any_float_value(list(linear.values()))
                or self._contains_any_float_value(list(quadratic.values()))):
            self._add_continuous_slack_var_quadratic_constraint(linear, quadratic, sense, rhs, name)
        # Else use an integer slack variable
        else:
            self._add_integer_slack_var_quadratic_constraint(linear, quadratic, sense, rhs, name)

    def _calc_linear_bounds(self, linear):
        lhs_lb, lhs_ub = 0, 0
        for var_name, v in linear.items():
            x = self._src.get_variable(var_name)
            lhs_lb += min(x.lowerbound * v, x.upperbound * v)
            lhs_ub += max(x.lowerbound * v, x.upperbound * v)
        return lhs_lb, lhs_ub

    def _calc_quadratic_bounds(self, linear, quadratic):
        lhs_lb, lhs_ub = 0, 0
        # Calculate the lowerbound and the upperbound of the linear part
        linear_lb, linear_ub = self._calc_linear_bounds(linear)
        lhs_lb += linear_lb
        lhs_ub += linear_ub

        # Calculate the lowerbound and the upperbound of the quadratic part
        for (name_i, name_j), v in quadratic.items():
            x = self._src.get_variable(name_i)
            y = self._src.get_variable(name_j)

            lhs_lb += min(
                x.lowerbound * y.lowerbound * v,
                x.lowerbound * y.upperbound * v,
                x.upperbound * y.lowerbound * v,
                x.upperbound * y.upperbound * v,
            )
            lhs_ub += max(
                x.lowerbound * y.lowerbound * v,
                x.lowerbound * y.upperbound * v,
                x.upperbound * y.lowerbound * v,
                x.upperbound * y.upperbound * v,
            )
        return lhs_lb, lhs_ub

    def interpret(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert a result of a converted problem into that of the original problem.

        Args:
            x: The result of the converted problem or the given result in case of FAILURE.

        Returns:
            The result of the original problem.
        """
        # convert back the optimization result into that of the original problem
        names = [var.name for var in self._dst.variables]

        # interpret slack variables
        sol = {name: x[i] for i, name in enumerate(names)}
        new_x = np.zeros(self._src.get_num_vars())
        for i, var in enumerate(self._src.variables):
            new_x[i] = sol[var.name]
        return new_x

    @staticmethod
    def _contains_any_float_value(values: List[Union[int, float]]) -> bool:
        """Check whether the list contains float or not.
        This method is used to check whether a constraint contain float coefficients or not.

        Args:
            values: Coefficients of the constraint

        Returns:
            bool: If the constraint contains float coefficients, this returns True, else False.
        """
        return any(isinstance(v, float) and not v.is_integer() for v in values)  # type: ignore

    @property
    def mode(self) -> str:
        """Returns the mode of the converter

        Returns:
            The mode of the converter used for additional slack variables
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Set a new mode for the converter

        Args:
            mode: The new mode for the converter
        """
        self._mode = mode

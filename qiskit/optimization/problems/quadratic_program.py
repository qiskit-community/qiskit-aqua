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

"""Quadratic Program."""

from typing import List, Union, Dict, Optional
from numpy import ndarray
from scipy.sparse import spmatrix

from qiskit.optimization import infinity, QiskitOptimizationError
from qiskit.optimization.problems.variable import Variable, VarType
from qiskit.optimization.problems.constraint import ConstraintSense
from qiskit.optimization.problems.linear_constraint import LinearConstraint


class QuadraticProgram:
    """Representation of a Quadratically Constrained Quadratic Program supporting inequality and
    equality constraints as well as continuous, binary, and integer variables.
    """

    def __init__(self, name: str = '') -> None:
        """Constructs a quadratic program.

        Args:
            name: The name of the quadratic program.
        """
        self._name = name

        self._variables: List[Variable] = []
        self._variables_index: Dict[str, int] = {}

        self._linear_constraints: List[LinearConstraint] = []
        self._linear_constraints_index: Dict[str, int] = {}

    @property
    def name(self) -> str:
        """Returns the name of the quadratic program.

        Returns:
            The name of the quadratic program.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the quadratic program.

        Args:
            name: The name of the quadratic program.
        """
        self._name = name

    @property
    def variables(self) -> List[Variable]:
        """Returns the list of variables of the quadratic program.

        Returns:
            List of variables.
        """
        return self._variables

    @property
    def variables_index(self) -> Dict[str, int]:
        """Returns the dictionary that maps the name of a variable to its index.

        Returns:
            The variable index dictionary.
        """
        return self._variables_index

    def _add_variables(self, name: Optional[str] = None, lowerbound: float = 0,
                       upperbound: float = infinity, vartype: VarType = VarType.continuous) -> None:
        """Checks whether a variable name is already taken and adds the variable to list and index
        if not.

        Args:
            name: The name of the variable.
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            vartype: The type of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already taken.

        """
        if name:
            if name in self._variables_index:
                raise QiskitOptimizationError("Variable name already exists!")
        else:
            k = self.get_num_vars()
            while 'x{}'.format(k) in self._variables_index:
                k += 1
            name = 'x{}'.format(k)
        self.variables_index[name] = len(self.variables)
        variable = Variable(self, name, lowerbound, upperbound, vartype)
        self.variables.append(variable)
        return variable

    def continuous_var(self, name: Optional[str] = None, lowerbound: float = 0,
                       upperbound: float = infinity) -> Variable:
        """Adds a continuous variable to the quadratic program.

        Args:
            name: The name of the variable.
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variables(name, lowerbound, upperbound, VarType.continuous)

    def binary_var(self, name: Optional[str] = None) -> Variable:
        """Adds a binary variable to the quadratic program.

        Args:
            name: The name of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variables(name, 0, 1, VarType.binary)

    def integer_var(self, name: Optional[str] = None, lowerbound: float = 0,
                    upperbound: float = infinity) -> Variable:
        """Adds an integer variable to the quadratic program.

        Args:
            name: The name of the variable.
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variables(name, lowerbound, upperbound, VarType.integer)

    def get_variable(self, i: Union[int, str]) -> Variable:
        """Returns a variable for a given name or index.

        Args:
            i: the index or name of the variable.

        Returns:
            The corresponding variable.
        """
        if isinstance(i, int):
            return self.variables[i]
        else:
            return self.variables[self._variables_index[i]]

    def get_num_vars(self, vartype: Optional[VarType] = None) -> int:
        """Returns the total number of variables or the number of variables of the specified type.

        Args:
            vartype: The type to be filtered on. All variables are counted if None.

        Returns:
            The total number of variables.
        """
        if vartype:
            return sum([variable.vartype == vartype for variable in self.variables])
        else:
            return len(self.variables)

    def get_num_continuous_vars(self) -> int:
        """Returns the total number of continuous variables.

        Returns:
            The total number of continuous variables.
        """
        return self.get_num_vars(VarType.continuous)

    def get_num_binary_vars(self) -> int:
        """Returns the total number of binary variables.

        Returns:
            The total number of binary variables.
        """
        return self.get_num_vars(VarType.binary)

    def get_num_integer_vars(self) -> int:
        """Returns the total number of integer variables.

        Returns:
            The total number of integer variables.
        """
        return self.get_num_vars(VarType.integer)

    @property
    def linear_constraints(self) -> List[LinearConstraint]:
        """Returns the list of linear constraints of the quadratic program.

        Returns:
            List of linear constraints.
        """
        return self._linear_constraints

    @property
    def linear_constraints_index(self) -> Dict[str, int]:
        """Returns the dictionary that maps the name of a linear constraint to its index.

        Returns:
            The linear constraint index dictionary.
        """
        return self._linear_constraints_index

    def _add_linear_constraint(self,
                               name: Optional[str] = None,
                               linear_coefficients: Union[ndarray, spmatrix, List[float],
                                                          Dict[Union[int, str], float]] = None,
                               sense: ConstraintSense = ConstraintSense.leq,
                               rhs: float = 0.0
                               ) -> None:
        """Checks whether a constraint name is already taken and adds the constraint to list and
        index if not.

        Args:
            name: The name of the constraint.
            linear_coefficients: The linear coefficients of the constraint.
            sense: The constraint sense.
            rhs: The right-hand-side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name is already taken.

        """
        if name:
            if name in self.linear_constraints_index:
                raise QiskitOptimizationError("Variable name already exists!")
        else:
            k = self.get_num_linear_constraints()
            while 'c{}'.format(k) in self.linear_constraints_index:
                k += 1
            name = 'c{}'.format(k)
        self.linear_constraints_index[name] = len(self.linear_constraints)
        if linear_coefficients is None:
            linear_coefficients = {}
        constraint = LinearConstraint(self, name, linear_coefficients, sense, rhs)
        self.linear_constraints.append(constraint)
        return constraint

    def linear_eq_constraint(self, name: Optional[str] = None,
                             linear_coefficients: Union[ndarray, spmatrix, List[float],
                                                        Dict[Union[int, str], float]] = None,
                             rhs: float = 0.0) -> LinearConstraint:
        """Adds a linear equality constraint to the quadratic program of the form:
            linear_coeffs * x == rhs.

        Args:
            name: The name of the constraint.
            linear_coefficients: The linear coefficients of the left-hand-side of the constraint.
            rhs: The right hand side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists.
        """
        return self._add_linear_constraint(name, linear_coefficients, ConstraintSense.eq, rhs)

    def linear_geq_constraint(self, name: Optional[str] = None,
                              linear_coefficients: Union[ndarray, spmatrix, List[float],
                                                         Dict[Union[int, str], float]] = None,
                              rhs: float = 0.0) -> LinearConstraint:
        """Adds a linear "greater-than-or-equal-to" (geq) constraint to the quadratic program
        of the form:
            linear_coeffs * x >= rhs.

        Args:
            name: The name of the constraint.
            linear_coefficients: The linear coefficients of the left-hand-side of the constraint.
            rhs: The right hand side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists.
        """
        return self._add_linear_constraint(name, linear_coefficients, ConstraintSense.geq, rhs)

    def linear_leq_constraint(self, name: Optional[str] = None,
                              linear_coefficients: Union[ndarray, spmatrix, List[float],
                                                         Dict[Union[int, str], float]] = None,
                              rhs: float = 0.0) -> LinearConstraint:
        """Adds a linear "less-than-or-equal-to" (leq) constraint to the quadratic program
        of the form:
            linear_coeffs * x <= rhs.

        Args:
            name: The name of the constraint.
            linear_coefficients: The linear coefficients of the left-hand-side of the constraint.
            rhs: The right hand side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists.
        """
        return self._add_linear_constraint(name, linear_coefficients, ConstraintSense.leq, rhs)

    def get_linear_constraint(self, i: Union[int, str]) -> LinearConstraint:
        """Returns a linear constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.
        """
        if isinstance(i, int):
            return self.linear_constraints[i]
        else:
            return self.linear_constraints[self._linear_constraints_index[i]]

    def get_num_linear_constraints(self) -> int:
        """Returns the number of linear constraints.

        Returns:
            The number of linear constraints.
        """
        return len(self.linear_constraints)

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

from typing import List, Union, Dict, Optional, Tuple

from docplex.mp.linear import Var
from docplex.mp.model import Model
from numpy import ndarray
from scipy.sparse import spmatrix

from qiskit.optimization import infinity, QiskitOptimizationError
from qiskit.optimization.problems.constraint import ConstraintSense
from qiskit.optimization.problems.linear_constraint import LinearConstraint
from qiskit.optimization.problems.quadratic_constraint import QuadraticConstraint
from qiskit.optimization.problems.quadratic_objective import QuadraticObjective, ObjSense
from qiskit.optimization.problems.variable import Variable, VarType


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

        self._quadratic_constraints: List[QuadraticConstraint] = []
        self._quadratic_constraints_index: Dict[str, int] = {}

        self._objective = QuadraticObjective(self)

    def clear(self) -> None:
        """Clears the quadratic program, i.e., deletes all variables, constraints, the
        objective function as well as the name.
        """
        self._name = ''

        self._variables: List[Variable] = []
        self._variables_index: Dict[str, int] = {}

        self._linear_constraints: List[LinearConstraint] = []
        self._linear_constraints_index: Dict[str, int] = {}

        self._quadratic_constraints: List[QuadraticConstraint] = []
        self._quadratic_constraints_index: Dict[str, int] = {}

        self._objective = QuadraticObjective(self)

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

    def _add_variable(self, name: Optional[str] = None, lowerbound: float = 0,
                      upperbound: float = infinity,
                      vartype: VarType = VarType.continuous) -> Variable:
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
                raise QiskitOptimizationError("Variable name already exists: {}".format(name))
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
        return self._add_variable(name, lowerbound, upperbound, VarType.continuous)

    def binary_var(self, name: Optional[str] = None) -> Variable:
        """Adds a binary variable to the quadratic program.

        Args:
            name: The name of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(name, 0, 1, VarType.binary)

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
        return self._add_variable(name, lowerbound, upperbound, VarType.integer)

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
            return sum(variable.vartype == vartype for variable in self.variables)
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

    def linear_constraint(self, name: Optional[str] = None,
                          coefficients: Union[ndarray, spmatrix, List[float],
                                              Dict[Union[int, str], float]] = None,
                          sense: Union[str, ConstraintSense] = '<=',
                          rhs: float = 0.0) -> LinearConstraint:
        """Adds a linear equality constraint to the quadratic program of the form:
            linear_coeffs * x sense rhs.

        Args:
            name: The name of the constraint.
            coefficients: The linear coefficients of the left-hand-side of the constraint.
            sense: The sense of the constraint,
              - '==', '=', 'E', and 'EQ' denote 'equal to'.
              - '>=', '>', 'G', and 'GE' denote 'greater-than-or-equal-to'.
              - '<=', '<', 'L', and 'LE' denote 'less-than-or-equal-to'.
            rhs: The right hand side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists or the sense is not
                valid.
        """
        if name:
            if name in self.linear_constraints_index:
                raise QiskitOptimizationError(
                    "Linear constraint's name already exists: {}".format(name))
        else:
            k = self.get_num_linear_constraints()
            while 'c{}'.format(k) in self.linear_constraints_index:
                k += 1
            name = 'c{}'.format(k)
        self.linear_constraints_index[name] = len(self.linear_constraints)
        if coefficients is None:
            coefficients = {}
        constraint = LinearConstraint(self, name, coefficients, ConstraintSense.convert(sense), rhs)
        self.linear_constraints.append(constraint)
        return constraint

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

    @property
    def quadratic_constraints(self) -> List[QuadraticConstraint]:
        """Returns the list of quadratic constraints of the quadratic program.

        Returns:
            List of quadratic constraints.
        """
        return self._quadratic_constraints

    @property
    def quadratic_constraints_index(self) -> Dict[str, int]:
        """Returns the dictionary that maps the name of a quadratic constraint to its index.

        Returns:
            The quadratic constraint index dictionary.
        """
        return self._quadratic_constraints_index

    def quadratic_constraint(self, name: Optional[str] = None,
                             linear_coefficients: Union[ndarray, spmatrix, List[float],
                                                        Dict[Union[int, str], float]] = None,
                             quadratic_coefficients: Union[ndarray, spmatrix,
                                                           List[List[float]],
                                                           Dict[
                                                               Tuple[Union[int, str],
                                                                     Union[int, str]],
                                                               float]] = None,
                             sense: Union[str, ConstraintSense] = '<=',
                             rhs: float = 0.0) -> QuadraticConstraint:
        """Adds a quadratic equality constraint to the quadratic program of the form:
            x * Q * x <= rhs.

        Args:
            name: The name of the constraint.
            linear_coefficients: The linear coefficients of the constraint.
            quadratic_coefficients: The quadratic coefficients of the constraint.
            sense: The sense of the constraint,
              - '==', '=', 'E', and 'EQ' denote 'equal to'.
              - '>=', '>', 'G', and 'GE' denote 'greater-than-or-equal-to'.
              - '<=', '<', 'L', and 'LE' denote 'less-than-or-equal-to'.
            rhs: The right hand side of the constraint.

        Returns:
            The added constraint.

        Raises:
            QiskitOptimizationError: if the constraint name already exists.
        """
        if name:
            if name in self.quadratic_constraints_index:
                raise QiskitOptimizationError(
                    "Quadratic constraint name already exists: {}".format(name))
        else:
            k = self.get_num_quadratic_constraints()
            while 'q{}'.format(k) in self.quadratic_constraints_index:
                k += 1
            name = 'q{}'.format(k)
        self.quadratic_constraints_index[name] = len(self.quadratic_constraints)
        if linear_coefficients is None:
            linear_coefficients = {}
        if quadratic_coefficients is None:
            quadratic_coefficients = {}
        constraint = QuadraticConstraint(self, name, linear_coefficients, quadratic_coefficients,
                                         ConstraintSense.convert(sense), rhs)
        self.quadratic_constraints.append(constraint)
        return constraint

    def get_quadratic_constraint(self, i: Union[int, str]) -> QuadraticConstraint:
        """Returns a quadratic constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.
        """
        if isinstance(i, int):
            return self.quadratic_constraints[i]
        else:
            return self.quadratic_constraints[self._quadratic_constraints_index[i]]

    def get_num_quadratic_constraints(self) -> int:
        """Returns the number of quadratic constraints.

        Returns:
            The number of quadratic constraints.
        """
        return len(self.quadratic_constraints)

    @property
    def objective(self) -> QuadraticObjective:
        """Returns the quadratic objective.

        Returns:
            The quadratic objective.
        """
        return self._objective

    def minimize(self,
                 constant: float = 0.0,
                 linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]] = None,
                 quadratic: Union[ndarray, spmatrix, List[List[float]],
                                  Dict[Tuple[Union[int, str], Union[int, str]], float]] = None
                 ) -> None:
        """Sets a quadratic objective to be minimized.

        Args:
            constant: the constant offset of the objective.
            linear: the coefficients of the linear part of the objective.
            quadratic: the coefficients of the quadratic part of the objective.

        Returns:
            The created quadratic objective.
        """
        self._objective = QuadraticObjective(self, constant, linear, quadratic, ObjSense.minimize)

    def maximize(self,
                 constant: float = 0.0,
                 linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]] = None,
                 quadratic: Union[ndarray, spmatrix, List[List[float]],
                                  Dict[Tuple[Union[int, str], Union[int, str]], float]] = None
                 ) -> None:
        """Sets a quadratic objective to be maximized.

        Args:
            constant: the constant offset of the objective.
            linear: the coefficients of the linear part of the objective.
            quadratic: the coefficients of the quadratic part of the objective.

        Returns:
            The created quadratic objective.
        """
        self._objective = QuadraticObjective(self, constant, linear, quadratic, ObjSense.maximize)

    def from_docplex(self, model: Model) -> None:
        """Loads this quadratic program from a docplex model

        Args:
            model: The docplex model to be loaded.

        Raises:
            QiskitOptimizationError: if the model contains unsupported elements.
        """

        # clear current problem
        self.clear()

        # get name
        self.name = model.name

        # get variables
        for x in model.iter_variables():
            if x.get_vartype().one_letter_symbol() == 'C':
                self.continuous_var(x.name, x.lb, x.ub)
            elif x.get_vartype().one_letter_symbol() == 'B':
                self.binary_var(x.name)
            elif x.get_vartype().one_letter_symbol() == 'I':
                self.integer_var(x.name, x.lb, x.ub)
            else:
                raise QiskitOptimizationError("Unsupported variable type!")

        # objective sense
        minimize = model.objective_sense.is_minimize()

        # get objective offset
        constant = model.objective_expr.constant

        # get linear part of objective
        linear_part = model.objective_expr.get_linear_part()
        linear = {}
        for x in linear_part.iter_variables():
            linear[x.name] = linear_part.get_coef(x)

        # get quadratic part of objective
        quadratic = {}
        for quad_triplet in model.objective_expr.generate_quad_triplets():
            i = quad_triplet[0].name
            j = quad_triplet[1].name
            v = quad_triplet[2]
            quadratic[i, j] = v

        # set objective
        if minimize:
            self.minimize(constant, linear, quadratic)
        else:
            self.maximize(constant, linear, quadratic)

        # get linear constraints
        for i in range(model.number_of_linear_constraints):
            constraint = model.get_constraint_by_index(i)
            name = constraint.name
            sense = constraint.sense

            rhs = 0
            if not isinstance(constraint.lhs, Var):
                rhs -= constraint.lhs.constant
            if not isinstance(constraint.rhs, Var):
                rhs += constraint.rhs.constant

            lhs = {}
            for x in constraint.iter_net_linear_coefs():
                lhs[x[0].name] = x[1]

            if sense == sense.EQ:
                self.linear_constraint(name, lhs, '==', rhs)
            elif sense == sense.GE:
                self.linear_constraint(name, lhs, '>=', rhs)
            elif sense == sense.LE:
                self.linear_constraint(name, lhs, '<=', rhs)
            else:
                raise QiskitOptimizationError("Unsupported constraint sense!")

        # get quadratic constraints
        for i in range(model.number_of_quadratic_constraints):
            constraint = model.get_quadratic_by_index(i)
            name = constraint.name
            sense = constraint.sense

            left_expr = constraint.get_left_expr()
            right_expr = constraint.get_right_expr()

            rhs = right_expr.constant - left_expr.constant
            linear = {}
            quadratic = {}

            if left_expr.is_quad_expr():
                for x in left_expr.linear_part.iter_variables():
                    linear[x.name] = left_expr.linear_part.get_coef(x)
                for quad_triplet in left_expr.iter_quad_triplets():
                    i = quad_triplet[0].name
                    j = quad_triplet[1].name
                    v = quad_triplet[2]
                    quadratic[i, j] = v
            else:
                for x in left_expr.iter_variables():
                    linear[x.name] = left_expr.get_coef(x)

            if right_expr.is_quad_expr():
                for x in right_expr.linear_part.iter_variables():
                    linear[x.name] = linear.get(x.name, 0.0) - right_expr.linear_part.get_coef(x)
                for quad_triplet in right_expr.iter_quad_triplets():
                    i = quad_triplet[0].name
                    j = quad_triplet[1].name
                    v = quad_triplet[2]
                    quadratic[i, j] = quadratic.get((i, j), 0.0) - v
            else:
                for x in right_expr.iter_variables():
                    linear[x.name] = linear.get(x.name, 0.0) - right_expr.get_coef(x)

            if sense == sense.EQ:
                self.quadratic_constraint(name, linear, quadratic, '==', rhs)
            elif sense == sense.GE:
                self.quadratic_constraint(name, linear, quadratic, '>=', rhs)
            elif sense == sense.LE:
                self.quadratic_constraint(name, linear, quadratic, '<=', rhs)
            else:
                raise QiskitOptimizationError("Unsupported constraint sense!")

    def to_docplex(self) -> Model:
        """Returns a docplex model corresponding to this quadratic program.

        Returns:
            The docplex model corresponding to this quadratic program.

        Raises:
            QiskitOptimizationError: if non-supported elements (should never happen).
        """

        # initialize model
        mdl = Model(self.name)

        # add variables
        var = {}
        for i, x in enumerate(self.variables):
            if x.vartype == VarType.continuous:
                var[i] = mdl.continuous_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            elif x.vartype == VarType.binary:
                var[i] = mdl.binary_var(name=x.name)
            elif x.vartype == VarType.integer:
                var[i] = mdl.integer_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            else:
                # should never happen
                raise QiskitOptimizationError('Unknown variable type: %s!' % x.vartype)

        # add objective
        objective = self.objective.constant
        for i, v in self.objective.linear.coefficients_as_dict().items():
            objective += v * var[i]
        for (i, j), v in self.objective.quadratic.coefficients_as_dict().items():
            objective += v * var[i] * var[j]
        if self.objective.sense == ObjSense.minimize:
            mdl.minimize(objective)
        else:
            mdl.maximize(objective)

        # add linear constraints
        for i, constraint in enumerate(self.linear_constraints):
            name = constraint.name
            rhs = constraint.rhs
            linear_expr = 0
            for j, v in constraint.linear.coefficients_as_dict().items():
                linear_expr += v * var[j]
            sense = constraint.sense
            if sense == ConstraintSense.eq:
                mdl.add_constraint(linear_expr == rhs, ctname=name)
            elif sense == ConstraintSense.geq:
                mdl.add_constraint(linear_expr >= rhs, ctname=name)
            elif sense == ConstraintSense.leq:
                mdl.add_constraint(linear_expr <= rhs, ctname=name)
            else:
                # should never happen
                raise QiskitOptimizationError('Unknown sense: %s!' % sense)

        # add quadratic constraints
        for i, constraint in enumerate(self.quadratic_constraints):
            name = constraint.name
            rhs = constraint.rhs
            quadratic_expr = 0
            for j, v in constraint.linear.coefficients_as_dict().items():
                quadratic_expr += v * var[j]
            for (j, k), v in constraint.quadratic.coefficients_as_dict().items():
                quadratic_expr += v * var[j] * var[k]
            sense = constraint.sense
            if sense == ConstraintSense.eq:
                mdl.add_constraint(quadratic_expr == rhs, ctname=name)
            elif sense == ConstraintSense.geq:
                mdl.add_constraint(quadratic_expr >= rhs, ctname=name)
            elif sense == ConstraintSense.leq:
                mdl.add_constraint(quadratic_expr <= rhs, ctname=name)
            else:
                # should never happen
                raise QiskitOptimizationError('Unknown sense: %s!' % sense)

        return mdl

    def pprint_as_string(self) -> str:
        """Pretty prints the quadratic program as a string.

        Returns:
            A string representing the quadratic program.
        """
        return self.to_docplex().pprint_as_string()

    def prettyprint(self, out: Optional[str] = None) -> None:
        """Pretty prints the quadratic program to a given output stream (None = default).

        Args:
            out: The output stream or file name to print to.
              if you specify a file name, the output file name is has '.mod' as suffix.
        """
        self.to_docplex().prettyprint(out)

    def print_as_lp_string(self) -> str:
        """Prints the quadratic program as a string of LP format.

        Returns:
            A string representing the quadratic program.
        """
        return self.to_docplex().export_as_lp_string()

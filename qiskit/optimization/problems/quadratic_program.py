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

from typing import cast, List, Union, Dict, Optional, Tuple
import logging
from collections import defaultdict
from enum import Enum
from math import fsum
import warnings

from docplex.mp.constr import (LinearConstraint as DocplexLinearConstraint,
                               QuadraticConstraint as DocplexQuadraticConstraint,
                               NotEqualConstraint)
from docplex.mp.linear import Var
from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader
from docplex.mp.quad import QuadExpr
from numpy import (ndarray, zeros, bool as nbool)
from scipy.sparse import spmatrix

from qiskit.aqua.operators import I, OperatorBase, PauliOp, WeightedPauliOperator, SummedOp, ListOp
from qiskit.quantum_info import Pauli
from .constraint import Constraint, ConstraintSense
from .linear_constraint import LinearConstraint
from .linear_expression import LinearExpression
from .quadratic_constraint import QuadraticConstraint
from .quadratic_expression import QuadraticExpression
from .quadratic_objective import QuadraticObjective
from .variable import Variable, VarType
from ..exceptions import QiskitOptimizationError
from ..infinity import INFINITY

logger = logging.getLogger(__name__)


class QuadraticProgramStatus(Enum):
    """Status of QuadraticProgram"""
    VALID = 0
    INFEASIBLE = 1


class QuadraticProgram:
    """Quadratically Constrained Quadratic Program representation.

    This representation supports inequality and equality constraints,
    as well as continuous, binary, and integer variables.
    """
    Status = QuadraticProgramStatus

    def __init__(self, name: str = '') -> None:
        """
        Args:
            name: The name of the quadratic program.
        """
        self._name = name
        self._status = QuadraticProgram.Status.VALID

        self._variables = []  # type: List[Variable]
        self._variables_index = {}  # type: Dict[str, int]

        self._linear_constraints = []  # type: List[LinearConstraint]
        self._linear_constraints_index = {}  # type: Dict[str, int]

        self._quadratic_constraints = []  # type: List[QuadraticConstraint]
        self._quadratic_constraints_index = {}  # type: Dict[str, int]

        self._objective = QuadraticObjective(self)

    def __repr__(self) -> str:
        return self.to_docplex().export_as_lp_string()

    def clear(self) -> None:
        """Clears the quadratic program, i.e., deletes all variables, constraints, the
        objective function as well as the name.
        """
        self._name = ''
        self._status = QuadraticProgram.Status.VALID

        self._variables.clear()
        self._variables_index.clear()

        self._linear_constraints.clear()
        self._linear_constraints_index.clear()

        self._quadratic_constraints.clear()
        self._quadratic_constraints_index.clear()

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
    def status(self) -> QuadraticProgramStatus:
        """Status of the quadratic program.
        It can be infeasible due to variable substitution.

        Returns:
            The status of the quadratic program
        """
        return self._status

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

    def _add_variable(self,
                      lowerbound: Union[float, int] = 0,
                      upperbound: Union[float, int] = INFINITY,
                      vartype: VarType = VarType.CONTINUOUS,
                      name: Optional[str] = None) -> Variable:
        """Checks whether a variable name is already taken and adds the variable to list and index
        if not.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            vartype: The type of the variable.
            name: The name of the variable.

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

    def continuous_var(self, lowerbound: Union[float, int] = 0,
                       upperbound: Union[float, int] = INFINITY,
                       name: Optional[str] = None) -> Variable:
        """Adds a continuous variable to the quadratic program.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.CONTINUOUS, name)

    def binary_var(self, name: Optional[str] = None) -> Variable:
        """Adds a binary variable to the quadratic program.

        Args:
            name: The name of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(0, 1, Variable.Type.BINARY, name)

    def integer_var(self, lowerbound: Union[float, int] = 0,
                    upperbound: Union[float, int] = INFINITY,
                    name: Optional[str] = None) -> Variable:
        """Adds an integer variable to the quadratic program.

        Args:
            lowerbound: The lowerbound of the variable.
            upperbound: The upperbound of the variable.
            name: The name of the variable.

        Returns:
            The added variable.

        Raises:
            QiskitOptimizationError: if the variable name is already occupied.
        """
        return self._add_variable(lowerbound, upperbound, Variable.Type.INTEGER, name)

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
            return sum(variable.vartype == vartype for variable in self._variables)
        else:
            return len(self._variables)

    def get_num_continuous_vars(self) -> int:
        """Returns the total number of continuous variables.

        Returns:
            The total number of continuous variables.
        """
        return self.get_num_vars(Variable.Type.CONTINUOUS)

    def get_num_binary_vars(self) -> int:
        """Returns the total number of binary variables.

        Returns:
            The total number of binary variables.
        """
        return self.get_num_vars(Variable.Type.BINARY)

    def get_num_integer_vars(self) -> int:
        """Returns the total number of integer variables.

        Returns:
            The total number of integer variables.
        """
        return self.get_num_vars(Variable.Type.INTEGER)

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

    def linear_constraint(self,
                          linear: Union[ndarray, spmatrix, List[float],
                                        Dict[Union[int, str], float]] = None,
                          sense: Union[str, ConstraintSense] = '<=',
                          rhs: float = 0.0, name: Optional[str] = None) -> LinearConstraint:
        """Adds a linear equality constraint to the quadratic program of the form:
            linear * x sense rhs.

        Args:
            linear: The linear coefficients of the left-hand-side of the constraint.
            sense: The sense of the constraint,
              - '==', '=', 'E', and 'EQ' denote 'equal to'.
              - '>=', '>', 'G', and 'GE' denote 'greater-than-or-equal-to'.
              - '<=', '<', 'L', and 'LE' denote 'less-than-or-equal-to'.
            rhs: The right hand side of the constraint.
            name: The name of the constraint.

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
        if linear is None:
            linear = {}
        constraint = LinearConstraint(self, name, linear, Constraint.Sense.convert(sense), rhs)
        self.linear_constraints.append(constraint)
        return constraint

    def get_linear_constraint(self, i: Union[int, str]) -> LinearConstraint:
        """Returns a linear constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._linear_constraints[i]
        else:
            return self._linear_constraints[self._linear_constraints_index[i]]

    def get_num_linear_constraints(self) -> int:
        """Returns the number of linear constraints.

        Returns:
            The number of linear constraints.
        """
        return len(self._linear_constraints)

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

    def quadratic_constraint(self,
                             linear: Union[ndarray, spmatrix, List[float],
                                           Dict[Union[int, str], float]] = None,
                             quadratic: Union[ndarray, spmatrix, List[List[float]],
                                              Dict[Tuple[Union[int, str],
                                                         Union[int, str]], float]] = None,
                             sense: Union[str, ConstraintSense] = '<=',
                             rhs: float = 0.0, name: Optional[str] = None) -> QuadraticConstraint:
        """Adds a quadratic equality constraint to the quadratic program of the form:
            x * Q * x <= rhs.

        Args:
            linear: The linear coefficients of the constraint.
            quadratic: The quadratic coefficients of the constraint.
            sense: The sense of the constraint,
              - '==', '=', 'E', and 'EQ' denote 'equal to'.
              - '>=', '>', 'G', and 'GE' denote 'greater-than-or-equal-to'.
              - '<=', '<', 'L', and 'LE' denote 'less-than-or-equal-to'.
            rhs: The right hand side of the constraint.
            name: The name of the constraint.

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
        if linear is None:
            linear = {}
        if quadratic is None:
            quadratic = {}
        constraint = QuadraticConstraint(self, name, linear, quadratic,
                                         Constraint.Sense.convert(sense), rhs)
        self.quadratic_constraints.append(constraint)
        return constraint

    def get_quadratic_constraint(self, i: Union[int, str]) -> QuadraticConstraint:
        """Returns a quadratic constraint for a given name or index.

        Args:
            i: the index or name of the constraint.

        Returns:
            The corresponding constraint.

        Raises:
            IndexError: if the index is out of the list size
            KeyError: if the name does not exist
        """
        if isinstance(i, int):
            return self._quadratic_constraints[i]
        else:
            return self._quadratic_constraints[self._quadratic_constraints_index[i]]

    def get_num_quadratic_constraints(self) -> int:
        """Returns the number of quadratic constraints.

        Returns:
            The number of quadratic constraints.
        """
        return len(self._quadratic_constraints)

    def remove_linear_constraint(self, i: Union[str, int]) -> None:
        """Remove a linear constraint

        Args:
            i: an index or a name of a linear constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._linear_constraints_index[i]
        del self._linear_constraints[i]
        self._linear_constraints_index = {cst.name: j for j, cst in
                                          enumerate(self._linear_constraints)}

    def remove_quadratic_constraint(self, i: Union[str, int]) -> None:
        """Remove a quadratic constraint

        Args:
            i: an index or a name of a quadratic constraint

        Raises:
            KeyError: if name does not exist
            IndexError: if index is out of range
        """
        if isinstance(i, str):
            i = self._quadratic_constraints_index[i]
        del self._quadratic_constraints[i]
        self._quadratic_constraints_index = {cst.name: j for j, cst in
                                             enumerate(self._quadratic_constraints)}

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
        self._objective = QuadraticObjective(self, constant, linear, quadratic,
                                             QuadraticObjective.Sense.MINIMIZE)

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
        self._objective = QuadraticObjective(self, constant, linear, quadratic,
                                             QuadraticObjective.Sense.MAXIMIZE)

    def from_docplex(self, model: Model) -> None:
        """Loads this quadratic program from a docplex model.

        Note that this supports only basic functions of docplex as follows:
        - quadratic objective function
        - linear / quadratic constraints
        - binary / integer / continuous variables

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
        # keep track of names separately, since docplex allows to have None names.
        var_names = {}
        for x in model.iter_variables():
            if x.get_vartype().one_letter_symbol() == 'C':
                x_new = self.continuous_var(x.lb, x.ub, x.name)
            elif x.get_vartype().one_letter_symbol() == 'B':
                x_new = self.binary_var(x.name)
            elif x.get_vartype().one_letter_symbol() == 'I':
                x_new = self.integer_var(x.lb, x.ub, x.name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported variable type: {} {}".format(x.name, x.vartype))
            var_names[x] = x_new.name

        # objective sense
        minimize = model.objective_sense.is_minimize()

        # make sure objective expression is linear or quadratic and not a variable
        if isinstance(model.objective_expr, Var):
            model.objective_expr = model.objective_expr + 0

        # get objective offset
        constant = model.objective_expr.constant

        # get linear part of objective
        linear = {}
        linear_part = model.objective_expr.get_linear_part()
        for x in linear_part.iter_variables():
            linear[var_names[x]] = linear_part.get_coef(x)

        # get quadratic part of objective
        quadratic = {}
        if isinstance(model.objective_expr, QuadExpr):
            for quad_triplet in model.objective_expr.generate_quad_triplets():
                i = var_names[quad_triplet[0]]
                j = var_names[quad_triplet[1]]
                v = quad_triplet[2]
                quadratic[i, j] = v

        # set objective
        if minimize:
            self.minimize(constant, linear, quadratic)
        else:
            self.maximize(constant, linear, quadratic)

        # get linear constraints
        for constraint in model.iter_constraints():
            if isinstance(constraint, DocplexQuadraticConstraint):
                # ignore quadratic constraints here and process them later
                continue
            if not isinstance(constraint, DocplexLinearConstraint) or \
                    isinstance(constraint, NotEqualConstraint):
                # If any constraint is not linear/quadratic constraints, it raises an error.
                # Notice that NotEqualConstraint is a subclass of Docplex's LinearConstraint,
                # but it cannot be handled by Aqua optimization.
                raise QiskitOptimizationError(
                    'Unsupported constraint: {}'.format(constraint))
            name = constraint.name
            sense = constraint.sense

            rhs = 0
            if not isinstance(constraint.lhs, Var):
                rhs -= constraint.lhs.constant
            if not isinstance(constraint.rhs, Var):
                rhs += constraint.rhs.constant

            lhs = {}
            for x in constraint.iter_net_linear_coefs():
                lhs[var_names[x[0]]] = x[1]

            if sense == sense.EQ:
                self.linear_constraint(lhs, '==', rhs, name)
            elif sense == sense.GE:
                self.linear_constraint(lhs, '>=', rhs, name)
            elif sense == sense.LE:
                self.linear_constraint(lhs, '<=', rhs, name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported constraint sense: {}".format(constraint))

        # get quadratic constraints
        for constraint in model.iter_quadratic_constraints():
            name = constraint.name
            sense = constraint.sense

            left_expr = constraint.get_left_expr()
            right_expr = constraint.get_right_expr()

            rhs = right_expr.constant - left_expr.constant
            linear = {}
            quadratic = {}

            if left_expr.is_quad_expr():
                for x in left_expr.linear_part.iter_variables():
                    linear[var_names[x]] = left_expr.linear_part.get_coef(x)
                for quad_triplet in left_expr.iter_quad_triplets():
                    i = var_names[quad_triplet[0]]
                    j = var_names[quad_triplet[1]]
                    v = quad_triplet[2]
                    quadratic[i, j] = v
            else:
                for x in left_expr.iter_variables():
                    linear[var_names[x]] = left_expr.get_coef(x)

            if right_expr.is_quad_expr():
                for x in right_expr.linear_part.iter_variables():
                    linear[var_names[x]] = linear.get(var_names[x], 0.0) - \
                        right_expr.linear_part.get_coef(x)
                for quad_triplet in right_expr.iter_quad_triplets():
                    i = var_names[quad_triplet[0]]
                    j = var_names[quad_triplet[1]]
                    v = quad_triplet[2]
                    quadratic[i, j] = quadratic.get((i, j), 0.0) - v
            else:
                for x in right_expr.iter_variables():
                    linear[var_names[x]] = linear.get(var_names[x], 0.0) - right_expr.get_coef(x)

            if sense == sense.EQ:
                self.quadratic_constraint(linear, quadratic, '==', rhs, name)
            elif sense == sense.GE:
                self.quadratic_constraint(linear, quadratic, '>=', rhs, name)
            elif sense == sense.LE:
                self.quadratic_constraint(linear, quadratic, '<=', rhs, name)
            else:
                raise QiskitOptimizationError(
                    "Unsupported constraint sense: {}".format(constraint))

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
        for idx, x in enumerate(self.variables):
            if x.vartype == Variable.Type.CONTINUOUS:
                var[idx] = mdl.continuous_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            elif x.vartype == Variable.Type.BINARY:
                var[idx] = mdl.binary_var(name=x.name)
            elif x.vartype == Variable.Type.INTEGER:
                var[idx] = mdl.integer_var(lb=x.lowerbound, ub=x.upperbound, name=x.name)
            else:
                # should never happen
                raise QiskitOptimizationError('Unsupported variable type: {}'.format(x.vartype))

        # add objective
        objective = self.objective.constant
        for i, v in self.objective.linear.to_dict().items():
            objective += v * var[cast(int, i)]
        for (i, j), v in self.objective.quadratic.to_dict().items():
            objective += v * var[cast(int, i)] * var[cast(int, j)]
        if self.objective.sense == QuadraticObjective.Sense.MINIMIZE:
            mdl.minimize(objective)
        else:
            mdl.maximize(objective)

        # add linear constraints
        for i, l_constraint in enumerate(self.linear_constraints):
            name = l_constraint.name
            rhs = l_constraint.rhs
            if rhs == 0 and l_constraint.linear.coefficients.nnz == 0:
                continue
            linear_expr = 0
            for j, v in l_constraint.linear.to_dict().items():
                linear_expr += v * var[cast(int, j)]
            sense = l_constraint.sense
            if sense == Constraint.Sense.EQ:
                mdl.add_constraint(linear_expr == rhs, ctname=name)
            elif sense == Constraint.Sense.GE:
                mdl.add_constraint(linear_expr >= rhs, ctname=name)
            elif sense == Constraint.Sense.LE:
                mdl.add_constraint(linear_expr <= rhs, ctname=name)
            else:
                # should never happen
                raise QiskitOptimizationError("Unsupported constraint sense: {}".format(sense))

        # add quadratic constraints
        for i, q_constraint in enumerate(self.quadratic_constraints):
            name = q_constraint.name
            rhs = q_constraint.rhs
            if rhs == 0 \
                    and q_constraint.linear.coefficients.nnz == 0 \
                    and q_constraint.quadratic.coefficients.nnz == 0:
                continue
            quadratic_expr = 0
            for j, v in q_constraint.linear.to_dict().items():
                quadratic_expr += v * var[cast(int, j)]
            for (j, k), v in q_constraint.quadratic.to_dict().items():
                quadratic_expr += v * var[cast(int, j)] * var[cast(int, k)]
            sense = q_constraint.sense
            if sense == Constraint.Sense.EQ:
                mdl.add_constraint(quadratic_expr == rhs, ctname=name)
            elif sense == Constraint.Sense.GE:
                mdl.add_constraint(quadratic_expr >= rhs, ctname=name)
            elif sense == Constraint.Sense.LE:
                mdl.add_constraint(quadratic_expr <= rhs, ctname=name)
            else:
                # should never happen
                raise QiskitOptimizationError("Unsupported constraint sense: {}".format(sense))

        return mdl

    def export_as_lp_string(self) -> str:
        """Returns the quadratic program as a string of LP format.

        Returns:
            A string representing the quadratic program.
        """
        return self.to_docplex().export_as_lp_string()

    def pprint_as_string(self) -> str:
        """Returns the quadratic program as a string in Docplex's pretty print format.
        Returns:
            A string representing the quadratic program.
        """
        warnings.warn("The pprint_as_string method is deprecated and will be "
                      "removed in a future release. Instead use the"
                      "to_docplex() method and run pprint_as_string() on that "
                      "output", DeprecationWarning)
        return self.to_docplex().pprint_as_string()

    def prettyprint(self, out: Optional[str] = None) -> None:
        """Pretty prints the quadratic program to a given output stream (None = default).

        Args:
            out: The output stream or file name to print to.
              if you specify a file name, the output file name is has '.mod' as suffix.
        """
        warnings.warn("The prettyprint method is deprecated and will be "
                      "removed in a future release. Instead use the"
                      "to_docplex() method and run prettyprint() on that "
                      "output", DeprecationWarning)
        self.to_docplex().prettyprint(out)

    def read_from_lp_file(self, filename: str) -> None:
        """Loads the quadratic program from a LP file.

        Args:
            filename: The filename of the file to be loaded.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If CPLEX is not installed.

        Note:
            This method requires CPLEX to be installed and present in ``PYTHONPATH``.
        """
        try:
            import cplex  # pylint: disable=unused-import
        except ImportError:
            raise RuntimeError('The QuadraticProgram.read_from_lp_file method requires CPLEX to '
                               'be installed, but CPLEX could not be found.')

        def _parse_problem_name(filename: str) -> str:
            # Because docplex model reader uses the base name as model name,
            # we parse the model name in the LP file manually.
            # https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model_reader.html
            prefix = '\\Problem name:'
            model_name = ''
            with open(filename) as file:
                for line in file:
                    if line.startswith(prefix):
                        model_name = line[len(prefix):].strip()
                    if not line.startswith('\\'):
                        break
            return model_name

        model_reader = ModelReader()
        model = model_reader.read(filename, model_name=_parse_problem_name(filename))
        self.from_docplex(model)

    def write_to_lp_file(self, filename: str) -> None:
        """Writes the quadratic program to an LP file.

        Args:
            filename: The filename of the file the model is written to.
              If filename is a directory, file name 'my_problem.lp' is appended.
              If filename does not end with '.lp', suffix '.lp' is appended.

        Raises:
            OSError: If this cannot open a file.
            DOcplexException: If filename is an empty string
        """
        self.to_docplex().export_as_lp(filename)

    def substitute_variables(
            self, constants: Optional[Dict[Union[str, int], float]] = None,
            variables: Optional[Dict[Union[str, int], Tuple[Union[str, int], float]]] = None) \
            -> 'QuadraticProgram':
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                e.g., {'x': 2} means 'x' is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., {'x': ('y', 2)} means 'x' is substituted with 'y' * 2

        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, `QuadraticProgram.status` is still
            `QuadraticProgram.Status.VALIAD`.
            Otherwise, it gets `QuadraticProgram.Status.INFEASIBLE`.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.
                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """
        return SubstituteVariables().substitute_variables(self, constants, variables)

    def to_ising(self) -> Tuple[OperatorBase, float]:
        """Return the Ising Hamiltonian of this problem.

        Returns:
            qubit_op: The qubit operator for the problem
            offset: The constant value in the Ising Hamiltonian.

        Raises:
            QiskitOptimizationError: If a variable type is not binary.
            QiskitOptimizationError: If constraints exist in the problem.
        """
        # if problem has variables that are not binary, raise an error
        if self.get_num_vars() > self.get_num_binary_vars():
            raise QiskitOptimizationError('The type of variable must be a binary variable. '
                                          'Use a QuadraticProgramToQubo converter to convert '
                                          'integer variables to binary variables. '
                                          'If the problem contains continuous variables, '
                                          'currently we can not apply VQE/QAOA directly. '
                                          'you might want to use an ADMM optimizer '
                                          'for the problem. ')

        # if constraints exist, raise an error
        if self.linear_constraints \
                or self.quadratic_constraints:
            raise QiskitOptimizationError('An constraint exists. '
                                          'The method supports only model with no constraints. '
                                          'Use a QuadraticProgramToQubo converter. '
                                          'It converts inequality constraints to equality '
                                          'constraints, and then, it converters equality '
                                          'constraints to penalty terms of the object function.')

        # initialize Hamiltonian.
        num_nodes = self.get_num_vars()
        pauli_list = []
        offset = 0
        zero = zeros(num_nodes, dtype=nbool)

        # set a sign corresponding to a maximized or minimized problem.
        # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
        sense = self.objective.sense.value

        # convert a constant part of the object function into Hamiltonian.
        offset += self.objective.constant * sense

        # convert linear parts of the object function into Hamiltonian.
        for idx, coef in self.objective.linear.to_dict().items():
            z_p = zeros(num_nodes, dtype=nbool)
            weight = coef * sense / 2
            z_p[idx] = True

            pauli_list.append([-weight, Pauli(z_p, zero)])
            offset += weight

        # convert quadratic parts of the object function into Hamiltonian.
        # first merge coefficients (i, j) and (j, i)
        coeffs = {}  # type: Dict
        for (i, j), coeff in self.objective.quadratic.to_dict().items():
            if j < i:  # type: ignore
                coeffs[(j, i)] = coeffs.get((j, i), 0.0) + coeff
            else:
                coeffs[(i, j)] = coeffs.get((i, j), 0.0) + coeff

        # create Pauli terms
        for (i, j), coeff in coeffs.items():

            weight = coeff * sense / 4

            if i == j:
                offset += weight
            else:
                z_p = zeros(num_nodes, dtype=nbool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([weight, Pauli(z_p, zero)])

            z_p = zeros(num_nodes, dtype=nbool)
            z_p[i] = True
            pauli_list.append([-weight, Pauli(z_p, zero)])

            z_p = zeros(num_nodes, dtype=nbool)
            z_p[j] = True
            pauli_list.append([-weight, Pauli(z_p, zero)])

            offset += weight

        # Remove paulis whose coefficients are zeros.
        qubit_op = sum(PauliOp(pauli, coeff=coeff) for coeff, pauli in pauli_list)

        # qubit_op could be the integer 0, in this case return an identity operator of
        # appropriate size
        if isinstance(qubit_op, OperatorBase):
            qubit_op = qubit_op.reduce()
        else:
            qubit_op = I ^ num_nodes

        return qubit_op, offset

    def from_ising(self,
                   qubit_op: Union[OperatorBase, WeightedPauliOperator],
                   offset: float = 0.0, linear: bool = False) -> None:
        r"""Create a quadratic program from a qubit operator and a shift value.

        Args:
            qubit_op: The qubit operator of the problem.
            offset: The constant value in the Ising Hamiltonian.
            linear: If linear is True, :math:`x^2` is treated as a linear term
                since :math:`x^2 = x` for :math:`x \in \{0,1\}`.
                Else, :math:`x^2` is treat as a quadratic term.
                The default value is False.

        Raises:
            QiskitOptimizationError: If there are Pauli Xs in any Pauli term
            QiskitOptimizationError: If there are more than 2 Pauli Zs in any Pauli term
            NotImplementedError: If the input operator is a ListOp
        """
        if isinstance(qubit_op, WeightedPauliOperator):
            qubit_op = qubit_op.to_opflow()

        # No support for ListOp yet, this can be added in future
        # pylint: disable=unidiomatic-typecheck
        if type(qubit_op) == ListOp:
            raise NotImplementedError(
                'Conversion of a ListOp is not supported, convert each '
                'operator in the ListOp separately.'
            )

        # add binary variables
        for i in range(qubit_op.num_qubits):
            self.binary_var(name='x_{0}'.format(i))

        # Create a QUBO matrix
        # The Qubo matrix is an upper triangular matrix.
        # Diagonal elements in the QUBO matrix are for linear terms of the qubit operator.
        # The other elements in the QUBO matrix are for quadratic terms of the qubit operator.
        qubo_matrix = zeros((qubit_op.num_qubits, qubit_op.num_qubits))

        if not isinstance(qubit_op, SummedOp):
            oplist = [qubit_op.to_pauli_op()]
        else:
            oplist = qubit_op.to_pauli_op().oplist

        for pauli_op in oplist:
            pauli = pauli_op.primitive
            coeff = pauli_op.coeff
            # Count the number of Pauli Zs in a Pauli term
            lst_z = pauli.z.tolist()
            z_index = [i for i, z in enumerate(lst_z) if z is True]
            num_z = len(z_index)

            # Add its weight of the Pauli term to the corresponding element of QUBO matrix
            if num_z == 1:
                qubo_matrix[z_index[0], z_index[0]] = coeff.real
            elif num_z == 2:
                qubo_matrix[z_index[0], z_index[1]] = coeff.real
            else:
                raise QiskitOptimizationError(
                    'There are more than 2 Pauli Zs in the Pauli term {}'.format(pauli.z)
                )

            # If there are Pauli Xs in the Pauli term, raise an error
            lst_x = pauli.x.tolist()
            x_index = [i for i, x in enumerate(lst_x) if x is True]
            if len(x_index) > 0:
                raise QiskitOptimizationError('Pauli Xs exist in the Pauli {}'.format(pauli.x))

        # Initialize dicts for linear terms and quadratic terms
        linear_terms = {}
        quadratic_terms = {}

        # For quadratic pauli terms of operator
        # x_i * x_ j = (1 - Z_i - Z_j + Z_i * Z_j)/4
        for i, row in enumerate(qubo_matrix):
            for j, weight in enumerate(row):
                # Focus on the upper triangular matrix
                if j <= i:
                    continue
                # Add a quadratic term to the object function of `QuadraticProgram`
                # The coefficient of the quadratic term in `QuadraticProgram` is
                # 4 * weight of the pauli
                coef = weight * 4
                quadratic_terms[i, j] = coef
                # Sub the weight of the quadratic pauli term from the QUBO matrix
                qubo_matrix[i, j] -= weight
                # Sub the weight of the linear pauli term from the QUBO matrix
                qubo_matrix[i, i] += weight
                qubo_matrix[j, j] += weight
                # Sub the weight from offset
                offset -= weight

        # After processing quadratic pauli terms, only linear paulis are left
        # x_i = (1 - Z_i)/2
        for i in range(qubit_op.num_qubits):
            weight = qubo_matrix[i, i]
            # Add a linear term to the object function of `QuadraticProgram`
            # The coefficient of the linear term in `QuadraticProgram` is
            # 2 * weight of the pauli
            coef = weight * 2
            if linear:
                # If the linear option is True, add it into linear_terms
                linear_terms[i] = -coef
            else:
                # Else, add it into quadratic_terms as a diagonal element.
                quadratic_terms[i, i] = -coef
            # Sub the weight of the linear pauli term from the QUBO matrix
            qubo_matrix[i, i] -= weight
            offset += weight

        # Set the objective function
        self.minimize(constant=offset, linear=linear_terms, quadratic=quadratic_terms)
        offset -= offset


class SubstituteVariables:
    """A class to substitute variables of an optimization problem with constants for other
    variables"""

    CONST = '__CONSTANT__'

    def __init__(self):
        self._src = None  # type: Optional[QuadraticProgram]
        self._dst = None  # type: Optional[QuadraticProgram]
        self._subs = {}  # type: Dict[Union[int, str], Tuple[str, float]]

    def substitute_variables(
            self, src: QuadraticProgram,
            constants: Optional[Dict[Union[str, int], float]] = None,
            variables: Optional[Dict[Union[str, int], Tuple[Union[str, int], float]]] = None) \
            -> QuadraticProgram:
        """Substitutes variables with constants or other variables.

        Args:
            src: a quadratic program to be substituted.

            constants: replace variable by constant
                e.g., {'x': 2} means 'x' is substituted with 2

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                e.g., {'x': ('y', 2)} means 'x' is substituted with 'y' * 2

        Returns:
            An optimization problem by substituting variables with constants or other variables.
            If the substitution is valid, `QuadraticProgram.status` is still
            `QuadraticProgram.Status.VALIAD`.
            Otherwise, it gets `QuadraticProgram.Status.INFEASIBLE`.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.
                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """
        self._src = src
        self._dst = QuadraticProgram(src.name)
        self._subs_dict(constants, variables)
        results = [
            self._variables(),
            self._objective(),
            self._linear_constraints(),
            self._quadratic_constraints(),
        ]
        if any(not r for r in results):
            self._dst._status = QuadraticProgram.Status.INFEASIBLE
        return self._dst

    @staticmethod
    def _feasible(sense: ConstraintSense, rhs: float) -> bool:
        """Checks feasibility of the following condition
            0 `sense` rhs
        """
        # I use the following pylint option because `rhs` should come to right
        # pylint: disable=misplaced-comparison-constant
        if sense == Constraint.Sense.EQ:
            if 0 == rhs:
                return True
        elif sense == Constraint.Sense.LE:
            if 0 <= rhs:
                return True
        elif sense == Constraint.Sense.GE:
            if 0 >= rhs:
                return True
        return False

    @staticmethod
    def _replace_dict_keys_with_names(op, dic):
        key = []
        val = []
        for k in sorted(dic.keys()):
            key.append(op.variables.get_names(k))
            val.append(dic[k])
        return key, val

    def _subs_dict(self, constants, variables):
        # guarantee that there is no overlap between variables to be replaced and combine input
        subs = {}  # type: Dict[Union[int, str], Tuple[str, float]]
        if constants is not None:
            for i, v in constants.items():
                # substitute i <- v
                i_2 = self._src.get_variable(i).name
                if i_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable twice: {} <- {}'.format(i, v))
                subs[i_2] = (self.CONST, v)

        if variables is not None:
            for i, (j, v) in variables.items():
                if v == 0:
                    raise QiskitOptimizationError(
                        'coefficient must be non-zero: {} {} {}'.format(i, j, v))
                # substitute i <- j * v
                i_2 = self._src.get_variable(i).name
                j_2 = self._src.get_variable(j).name
                if i_2 == j_2:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable: {} <- {} {}'.format(i, j, v))
                if i_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute the same variable twice: {} <- {} {}'.format(i, j, v))
                if j_2 in subs:
                    raise QiskitOptimizationError(
                        'Cannot substitute by variable that gets substituted itself: '
                        '{} <- {} {}'.format(i, j, v))
                subs[i_2] = (j_2, v)

        self._subs = subs

    def _variables(self) -> bool:
        # copy variables that are not replaced
        feasible = True
        for var in self._src.variables:
            name = var.name
            vartype = var.vartype
            lowerbound = var.lowerbound
            upperbound = var.upperbound
            if name not in self._subs:
                self._dst._add_variable(lowerbound, upperbound, vartype, name)

        for i, (j, v) in self._subs.items():
            lb_i = self._src.get_variable(i).lowerbound
            ub_i = self._src.get_variable(i).upperbound
            if j == self.CONST:
                if not lb_i <= v <= ub_i:
                    logger.warning(
                        'Infeasible substitution for variable: %s', i)
                    feasible = False
            else:
                # substitute i <- j * v
                # lb_i <= i <= ub_i  -->  lb_i / v <= j <= ub_i / v if v > 0
                #                         ub_i / v <= j <= lb_i / v if v < 0
                if v == 0:
                    raise QiskitOptimizationError(
                        'Coefficient of variable substitution should be nonzero: '
                        '{} {} {}'.format(i, j, v))
                if abs(lb_i) < INFINITY:
                    new_lb_i = lb_i / v
                else:
                    new_lb_i = lb_i if v > 0 else -lb_i
                if abs(ub_i) < INFINITY:
                    new_ub_i = ub_i / v
                else:
                    new_ub_i = ub_i if v > 0 else -ub_i
                var_j = self._dst.get_variable(j)
                lb_j = var_j.lowerbound
                ub_j = var_j.upperbound
                if v > 0:
                    var_j.lowerbound = max(lb_j, new_lb_i)
                    var_j.upperbound = min(ub_j, new_ub_i)
                else:
                    var_j.lowerbound = max(lb_j, new_ub_i)
                    var_j.upperbound = min(ub_j, new_lb_i)

        for var in self._dst.variables:
            if var.lowerbound > var.upperbound:
                logger.warning(
                    'Infeasible lower and upper bound: %s %f %f', var, var.lowerbound,
                    var.upperbound)
                feasible = False

        return feasible

    def _linear_expression(self, lin_expr: LinearExpression) \
            -> Tuple[List[float], LinearExpression]:
        const = []
        lin_dict = defaultdict(float)  # type: Dict[Union[int, str], float]
        for i, w_i in lin_expr.to_dict(use_name=True).items():
            repl_i = self._subs[i] if i in self._subs else (i, 1)
            prod = w_i * repl_i[1]
            if repl_i[0] == self.CONST:
                const.append(prod)
            else:
                k = repl_i[0]
                lin_dict[k] += prod
        new_lin = LinearExpression(quadratic_program=self._dst,
                                   coefficients=lin_dict if lin_dict else {})
        return const, new_lin

    def _quadratic_expression(self, quad_expr: QuadraticExpression) \
            -> Tuple[List[float], Optional[LinearExpression], Optional[QuadraticExpression]]:
        const = []
        lin_dict = defaultdict(float)  # type: Dict[Union[int, str], float]
        quad_dict = defaultdict(float)  # type: Dict[Tuple[Union[int, str], Union[int, str]], float]
        for (i, j), w_ij in quad_expr.to_dict(use_name=True).items():
            repl_i = self._subs[i] if i in self._subs else (i, 1)
            repl_j = self._subs[j] if j in self._subs else (j, 1)
            idx = tuple(x for x, _ in [repl_i, repl_j] if x != self.CONST)
            prod = w_ij * repl_i[1] * repl_j[1]
            if len(idx) == 2:
                quad_dict[idx] += prod  # type: ignore
            elif len(idx) == 1:
                lin_dict[idx[0]] += prod
            else:
                const.append(prod)
        new_lin = LinearExpression(quadratic_program=self._dst,
                                   coefficients=lin_dict if lin_dict else {})
        new_quad = QuadraticExpression(quadratic_program=self._dst,
                                       coefficients=quad_dict if quad_dict else {})
        return const, new_lin, new_quad

    def _objective(self) -> bool:
        obj = self._src.objective
        const1, lin1 = self._linear_expression(obj.linear)
        const2, lin2, quadratic = self._quadratic_expression(obj.quadratic)

        constant = fsum([obj.constant] + const1 + const2)
        linear = lin1.coefficients + lin2.coefficients
        if obj.sense == obj.sense.MINIMIZE:
            self._dst.minimize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        else:
            self._dst.maximize(constant=constant, linear=linear, quadratic=quadratic.coefficients)
        return True

    def _linear_constraints(self) -> bool:
        feasible = True
        for lin_cst in self._src.linear_constraints:
            constant, linear = self._linear_expression(lin_cst.linear)
            rhs = -fsum([-lin_cst.rhs] + constant)
            if linear.coefficients.nnz > 0:
                self._dst.linear_constraint(name=lin_cst.name, linear=linear.coefficients,
                                            sense=lin_cst.sense, rhs=rhs)
            else:
                if not self._feasible(lin_cst.sense, rhs):
                    logger.warning('constraint %s is infeasible due to substitution', lin_cst.name)
                    feasible = False
        return feasible

    def _quadratic_constraints(self) -> bool:
        feasible = True
        for quad_cst in self._src.quadratic_constraints:
            const1, lin1 = self._linear_expression(quad_cst.linear)
            const2, lin2, quadratic = self._quadratic_expression(quad_cst.quadratic)
            rhs = -fsum([-quad_cst.rhs] + const1 + const2)
            linear = lin1.coefficients + lin2.coefficients

            if quadratic.coefficients.nnz > 0:
                self._dst.quadratic_constraint(name=quad_cst.name, linear=linear,
                                               quadratic=quadratic.coefficients,
                                               sense=quad_cst.sense, rhs=rhs)
            elif linear.nnz > 0:
                name = quad_cst.name
                lin_names = set(lin.name for lin in self._dst.linear_constraints)
                while name in lin_names:
                    name = '_' + name
                self._dst.linear_constraint(name=name, linear=linear, sense=quad_cst.sense, rhs=rhs)
            else:
                if not self._feasible(quad_cst.sense, rhs):
                    logger.warning('constraint %s is infeasible due to substitution', quad_cst.name)
                    feasible = False

        return feasible

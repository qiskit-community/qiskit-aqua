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

"""Mixed integer quadratically constrained quadratic program"""

from enum import Enum
from math import fsum
from typing import Optional, Tuple
import logging

from docplex.mp.model import Model as DocplexModel

from qiskit.optimization.problems.linear_constraint import LinearConstraintInterface
from qiskit.optimization.problems.objective import ObjectiveInterface
from qiskit.optimization.problems.problem_type import ProblemType
from qiskit.optimization.problems.quadratic_constraint import QuadraticConstraintInterface
from qiskit.optimization.problems.variables import VariablesInterface
from qiskit.optimization.results.solution import SolutionInterface
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import Cplex, SparsePair
    from cplex import SparseTriple, infinity
    from cplex.exceptions import CplexSolverError
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


class OptimizationProblem:
    """A class encapsulating an optimization problem, modeled after Python CPLEX API.

    An instance of the OptimizationProblem class provides methods for creating,
    modifying, and querying an optimization problem, solving it, and
    querying aspects of the solution.
    """

    def __init__(self, file_name: Optional[str] = None):
        """Constructor of the OptimizationProblem class.

        The OptimizationProblem constructor accepts four types of argument lists.

        Args:
            file_name: read a model from a file.

        Raises:
            QiskitOptimizationError: if it cannot load a file.
            NameError: CPLEX is not installed.

        Examples:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        op is a new problem with no data

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem("filename")
        op is a new problem containing the data in filename.  If
        filename does not exist, an exception is raised.

        The OptimizationProblem object is a context manager and can be used, like so:

        >>> from qiskit.optimization import OptimizationProblem
        >>> with OptimizationProblem() as op:
        >>>     # do stuff
        >>>     op.solve()

        When the with block is finished, the end() method will be called automatically.
        """
        if not _HAS_CPLEX:
            raise NameError('CPLEX is not installed.')

        self._name = ''

        self.variables = VariablesInterface()

        # convert variable names into indices
        varindex = self.variables.get_indices

        self.linear_constraints = LinearConstraintInterface(varindex=varindex)
        self.quadratic_constraints = QuadraticConstraintInterface(varindex=varindex)
        self.objective = ObjectiveInterface(varindex=varindex)
        self.solution = SolutionInterface()
        self.problem_type = ProblemType()

        # None means it will be detected automatically
        self._problem_type = None

        self.substitution_status = SubstitutionStatus

        # read from file in case filename is given
        if file_name:
            try:
                self.read(file_name)
            except CplexSolverError:
                raise QiskitOptimizationError('Could not load file: {}'.format(file_name))

    def from_cplex(self, op: 'Cplex'):
        """Loads an optimization problem from a Cplex object

        Args:
            op: a Cplex object
        """
        # make sure the current problem is clean
        self.end()

        self.set_problem_name(op.get_problem_name())
        self.set_problem_type(op.get_problem_type())

        # Note: CPLEX raises a "Not a MIP (3003)"-error if there is no variable whose type is not
        # specified. As a workaround, we add an dummy variable with a type and then delete it.
        idx = op.variables.add(types='B')
        op.variables.delete(idx[0])

        # set variables (obj is set via objective interface)
        lowerbounds = op.variables.get_lower_bounds()
        upperbounds = op.variables.get_upper_bounds()
        types = op.variables.get_types()
        names = op.variables.get_names()
        self.variables.add(lb=lowerbounds, ub=upperbounds, types=types, names=names)

        # set objective function
        try:
            # if no name is set for objective function, CPLEX raises CplexSolverError
            obj_name = op.objective.get_name()
        except CplexSolverError:
            obj_name = ''
        self.objective.set_name(obj_name)
        self.objective.set_sense(op.objective.get_sense())
        self.objective.set_offset(op.objective.get_offset())
        self.objective.set_linear((i, v) for i, v in enumerate(op.objective.get_linear()))
        if op.objective.get_num_quadratic_nonzeros() > 0:
            self.objective.set_quadratic(op.objective.get_quadratic())

        # set linear constraints
        lin_expr = op.linear_constraints.get_rows()
        senses = op.linear_constraints.get_senses()
        rhs = op.linear_constraints.get_rhs()
        range_values = op.linear_constraints.get_range_values()
        names = op.linear_constraints.get_names()
        self.linear_constraints.add(
            lin_expr=lin_expr, senses=senses, rhs=rhs, range_values=range_values, names=names)

        # set quadratic constraints
        names = op.quadratic_constraints.get_names()
        senses = op.quadratic_constraints.get_senses()
        rhs = op.quadratic_constraints.get_rhs()
        lin_expr = op.quadratic_constraints.get_linear_components()
        quad_expr = op.quadratic_constraints.get_quadratic_components()
        for i in range(op.quadratic_constraints.get_num()):
            self.quadratic_constraints.add(
                lin_expr=lin_expr[i], quad_expr=quad_expr[i], sense=senses[i], rhs=rhs[i],
                name=names[i])

    def from_docplex(self, model: DocplexModel):
        """Loads an optimization problem from a Docplex model

        Args:
            model: Docplex model
        """
        cpl = model.get_cplex()
        self.from_cplex(cpl)
        # Docplex does not copy the model name. We need to do it manually.
        self.set_problem_name(model.get_name())

    def to_cplex(self) -> 'Cplex':
        """Converts the optimization problem into a Cplex object.

        Returns: Cplex object
        """

        # create a new CPLEX model
        op = Cplex()
        op.set_problem_name(self.get_problem_name())

        # problem type will be set automatically by CPLEX

        # set variables
        names = self.variables.get_names()
        lowerbounds = self.variables.get_lower_bounds()
        upperbounds = self.variables.get_upper_bounds()
        types = self.variables.get_types()
        op.variables.add(lb=lowerbounds, ub=upperbounds, types=types, names=names)

        # set objective function
        op.objective.set_name(self.objective.get_name())
        op.objective.set_sense(self.objective.get_sense())
        op.objective.set_offset(self.objective.get_offset())
        op.objective.set_linear((i, v) for i, v in self.objective.get_linear_dict().items())
        if self.objective.get_num_quadratic_nonzeros() > 0:
            op.objective.set_quadratic(self.objective.get_quadratic())

        # set linear constraints
        lin_expr = self.linear_constraints.get_rows()
        senses = self.linear_constraints.get_senses()
        rhs = self.linear_constraints.get_rhs()
        range_values = self.linear_constraints.get_range_values()
        names = self.linear_constraints.get_names()
        op.linear_constraints.add(
            lin_expr=lin_expr, senses=senses, rhs=rhs, range_values=range_values, names=names)

        # set quadratic constraints
        names = self.quadratic_constraints.get_names()
        senses = self.quadratic_constraints.get_senses()
        rhs = self.quadratic_constraints.get_rhs()
        lin_expr = self.quadratic_constraints.get_linear_components()
        quad_expr = self.quadratic_constraints.get_quadratic_components()
        for i in range(self.quadratic_constraints.get_num()):
            op.quadratic_constraints.add(
                lin_expr=lin_expr[i], quad_expr=quad_expr[i], sense=senses[i], rhs=rhs[i],
                name=names[i])
        return op

    def end(self):
        """Releases the OptimizationProblem object."""
        self._name = ''
        self.variables = VariablesInterface()
        varindex = self.variables.get_indices
        self.linear_constraints = LinearConstraintInterface(varindex=varindex)
        self.quadratic_constraints = QuadraticConstraintInterface(varindex=varindex)
        self.objective = ObjectiveInterface(varindex=varindex)
        self.solution = SolutionInterface()
        self._problem_type = None

    def __enter__(self) -> 'OptimizationProblem':
        """To implement a ContextManager, as in Cplex."""
        return self

    def __exit__(self, *exc) -> bool:
        """To implement a ContextManager, as in Cplex."""
        self.end()
        return False

    def read(self, filename: str, filetype: str = ""):
        """Reads a problem from file.

        The first argument is a string specifying the filename from
        which the problem will be read.

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> op.read("lpex.mps")
        """
        cplex = Cplex()
        cplex.read(filename, filetype)
        self.from_cplex(cplex)

    def write(self, filename: str, filetype: str = ""):
        """Writes a problem to file.

        The first argument is a string specifying the filename to
        which the problem will be written.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names=['x1', 'x2', 'x3'])
        >>> op.write("example.lp")
        """
        cplex = self.to_cplex()
        cplex.write(filename, filetype)

    def write_to_stream(self, stream: object, filetype: str = 'LP', comptype: str = ''):
        """Writes a problem to a file-like object in the given file format.

        The filetype argument can be any of "sav" (a binary format), "lp"
        (the default), "mps", "rew", "rlp", or "alp" (see `OptimizationProblem.write`
        for an explanation of these).

        If comptype is "bz2" (for BZip2) or "gz" (for GNU Zip), a
        compressed file is written.

        Raises:
            QiskitOptimizationError: if `stream` does not have methods `write` and `flush`.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names=['x1', 'x2', 'x3'])
        >>> class NoOpStream(object):
        ...     def __init__(self):
        ...         self.was_called = False
        ...     def write(self, bytes):
        ...         self.was_called = True
        ...         pass
        ...     def flush(self):
        ...         pass
        >>> stream = NoOpStream()
        >>> op.write_to_stream(stream)
        >>> stream.was_called
        True
        """
        if not hasattr(stream, 'write') or not callable(stream.write):
            raise QiskitOptimizationError("stream must have a write method")
        if not hasattr(stream, 'flush') or not callable(stream.flush):
            raise QiskitOptimizationError("stream must have a flush method")
        op = self.to_cplex()
        op.write_to_stream(stream, filetype, comptype)

    def write_as_string(self, filetype: str = 'LP', comptype: str = '') -> str:
        """Writes a problem as a string in the given file format.

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names=['x1', 'x2', 'x3'])
        >>> lp_str = op.write_as_string("lp")
        >>> len(lp_str) > 0
        True
        """
        op = self.to_cplex()
        return op.write_as_string(filetype, comptype)

    def get_problem_type(self) -> int:
        """Returns the problem type.

        The return value is an attribute of self.problem_type.

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> op.read("lpex.mps")
        >>> op.get_problem_type()
        0
        >>> op.problem_type[op.get_problem_type()]
        'LP'
        """
        if self._problem_type:
            return self._problem_type
        return self._detect_problem_type()

    def _detect_problem_type(self) -> int:
        typ = self.problem_type.LP
        if self.variables.get_num() > 0:
            typ = self.problem_type.MILP
        if self.objective.get_num_quadratic_nonzeros() > 0:
            typ = self.problem_type.MIQP
        if self.quadratic_constraints.get_num() > 0:
            typ = self.problem_type.MIQCP
        return typ

    def set_problem_type(self, problem_type):
        """Changes the problem type.

        If only one argument is given, that argument specifies the new
        problem type.  It must be one of the following:

        qiskit.optimization.problem_type.LP
        qiskit.optimization.problem_type.MILP
        qiskit.optimization.problem_type.fixed_MILP
        qiskit.optimization.problem_type.QP
        qiskit.optimization.problem_type.MIQP
        qiskit.optimization.problem_type.fixed_MIQP
        qiskit.optimization.problem_type.QCP
        qiskit.optimization.problem_type.MIQCP
        """
        self._problem_type = problem_type

    def solve(self):
        """Prints out a message to ask users to use `OptimizationAlgorithm`.
        Users need to apply one of `OptimiztionAlgorithm`s instead of this method.
        """
        logger.warning('`OptimizationProblem.solve` is intentionally empty.'
                       'You can solve it by applying `OptimizationAlgorithm.solve`.')

    def set_problem_name(self, name: str):
        """Sets the problem name"""
        self._name = name

    def get_problem_name(self) -> str:
        """Returns the problem name"""
        return self._name

    def substitute_variables(self, constants: Optional['SparsePair'] = None,
                             variables: Optional['SparseTriple'] = None) \
            -> Tuple['OptimizationProblem', 'SubstitutionStatus']:
        """Substitutes variables with constants or other variables.

        Args:
            constants: replace variable by constant
                i.e., SparsePair.ind (variable) -> SparsePair.val (constant)

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly. The lower and upper bounds are updated accordingly.
                i.e., SparseTriple.ind1 (variable)
                        -> SparseTriple.ind2 (variable) * SparseTriple.val (constant)

        Returns:
            An optimization problem by substituting variables and the status.
            If the resulting problem has no issue, the status is `success`.
            Otherwise, an empty problem and status `infeasible` are returned.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.
                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> from cplex import SparsePair, SparseTriple
        >>> op = OptimizationProblem()
        >>> op.variables.add(names=['x', 'y'], types='I'*2, lb=[-1]*2, ub=[2]*2)
        >>> op.objective.set_sense(op.objective.sense.minimize)
        >>> op.objective.set_linear([('x', 1), ('y', 2)])
        >>> op.linear_constraints.add(lin_expr=[(['x', 'y'], [1.0, -1.0])], senses=['L'], rhs=[1.0])
        >>> print(op.write_as_string())
        \\ENCODING=ISO-8859-1
        \\Problem name:

        Minimize
         obj1: x + 2 y
        Subject To
         c1: x - y <= 1
        Bounds
        -1 <= x <= 2
        -1 <= y <= 2
        Generals
         x  y
        End
        >>> # substitute x <- 2
        >>> op2, st = op.substitute_variables(constants=SparsePair(ind=['x'], val=[2]))
        >>> print(st)
        SubstitutionStatus.success
        >>> print(op2.write_as_string())
        \\ENCODING=ISO-8859-1
        \\Problem name:

        Minimize
         obj1: 2 y + 2
        Subject To
         c1: - y <= -1
        Bounds
        -1 <= y <= 2
        Generals
         y
        End
        >>> # substitute y <- -x
        >>> op3, st = op.substitute_variables(variables=SparseTriple(\
                                                ind1=['y'], ind2=['x'], val=[-1]))
        >>> print(st)
        SubstitutionStatus.success
        >>> print(op3.write_as_string())
        \\ENCODING=ISO-8859-1
        \\Problem name:

        Minimize
         obj1: - x
        Subject To
         c1: 2 x <= 1
        Bounds
        -1 <= x <= 1
        Generals
         x
        End
        """
        subs = SubstituteVariables()
        return subs.substitute_variables(src=self, constants=constants, variables=variables)


class SubstitutionStatus(Enum):
    """Status of `OptimizationProblem.substitute_variables`"""
    success = 1
    infeasible = 2


class SubstituteVariables:
    """A class to substitute variables of an optimization problem with constants for other
    variables"""

    CONST = -1

    def __init__(self):
        self._src: OptimizationProblem = None
        self._dst: OptimizationProblem = None
        self._subs = {}

    def substitute_variables(self, src: OptimizationProblem,
                             constants: Optional['SparsePair'] = None,
                             variables: Optional['SparseTriple'] = None) \
            -> Tuple[OptimizationProblem, SubstitutionStatus]:
        """Substitutes variables with constants or other variables.

        Args:
            src: an optimization problem whose variables will be substituted
            constants: replace variable by constant
                i.e., SparsePair.ind (variable) -> SparsePair.val (constant)

            variables: replace variables by weighted other variable
                need to copy everything using name reference to make sure that indices are matched
                correctly
                i.e., SparseTriple.ind1 (variable)
                        -> SparseTriple.ind2 (variable) * SparseTriple.val (constant)

        Returns:
            An optimization problem by substituting variables and the status.
            If the resulting problem has no issue, the status is `success`.
            Otherwise, an empty problem and status `infeasible` are returned.

        Raises:
            QiskitOptimizationError: if the substitution is invalid as follows.
                - Same variable is substituted multiple times.
                - Coefficient of variable substitution is zero.
        """

        self._src = src
        self._dst = OptimizationProblem()
        self._dst.set_problem_name(src.get_problem_name())
        # do not set problem type, then it detects its type automatically

        self._subs_dict(constants, variables)

        results = [
            self._variables(),
            self._objective(),
            self._linear_constraints(),
            self._quadratic_constraints(),
        ]
        if any(r == SubstitutionStatus.infeasible for r in results):
            ret = SubstitutionStatus.infeasible
        else:
            ret = SubstitutionStatus.success
        return self._dst, ret

    @staticmethod
    def _feasible(sense: str, rhs: float, range_value: float = 0) -> bool:
        """Checks feasibility of the following condition
            0 `sense` rhs
        """
        # I use the following pylint option because `rhs` should come to right
        # pylint: disable=misplaced-comparison-constant
        if sense == 'E':
            if 0 == rhs:
                return True
        elif sense == 'L':
            if 0 <= rhs:
                return True
        elif sense == 'G':
            if 0 >= rhs:
                return True
        else:  # sense == 'R'
            if range_value >= 0:
                if rhs <= 0 <= rhs + range_value:
                    return True
            else:
                if rhs + range_value <= 0 <= rhs:
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
        src = self._src

        # guarantee that there is no overlap between variables to be replaced and combine input
        subs = {}
        if constants is not None:
            if not isinstance(constants, SparsePair):
                raise QiskitOptimizationError(
                    'substitution with constant should be SparsePair: {}'.format(constants))
            for i, v in zip(constants.ind, constants.val):
                # substitute i <- v
                i_2 = src.variables.get_indices(i)
                if i_2 in subs:
                    raise QiskitOptimizationError(
                        'cannot substitute the same variable twice: {} <- {}'.format(i, v))
                subs[i_2] = (self.CONST, v)

        if variables is not None:
            if not isinstance(variables, SparseTriple):
                raise QiskitOptimizationError(
                    'substitution with variable should be SparseTriple: {}'.format(variables))
            for i, j, v in zip(variables.ind1, variables.ind2, variables.val):
                if v == 0:
                    raise QiskitOptimizationError(
                        'coefficient should not be zero: {} {} {}'.format(i, j, v))
                # substitute i <- j * v
                i_2 = src.variables.get_indices(i)
                j_2 = src.variables.get_indices(j)
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

    def _variables(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst
        subs = self._subs

        # copy variables that are not replaced
        for name, var_type, lowerbound, upperbound in zip(
                src.variables.get_names(),
                src.variables.get_types(),
                src.variables.get_lower_bounds(),
                src.variables.get_upper_bounds(),
        ):
            i = src.variables.get_indices(name)
            if i not in subs:
                dst.variables.add(lb=[lowerbound], ub=[upperbound], types=var_type, names=[name])

        for i, (j, v) in subs.items():
            var_i = src.variables.get_names(i)
            lb_i = src.variables.get_lower_bounds(i)
            ub_i = src.variables.get_upper_bounds(i)
            if j == self.CONST:
                if not lb_i <= v <= ub_i:
                    logger.warning(
                        'Infeasible substitution for variable: %s', var_i)
                    return SubstitutionStatus.infeasible
            else:
                # substitute i <- j * v
                # lb_i <= i <= ub_i  -->  lb_i / v <= j <= ub_i / v if v > 0
                #                         ub_i / v <= j <= lb_i / v if v < 0
                if v == 0:
                    raise QiskitOptimizationError(
                        'Coefficient of variable substitution should be nonzero: '
                        '{} {} {}'.format(i, j, v))
                var_j = src.variables.get_names(j)
                if abs(lb_i) < infinity:
                    new_lb_i = lb_i / v
                else:
                    new_lb_i = lb_i if v > 0 else -lb_i
                if abs(ub_i) < infinity:
                    new_ub_i = ub_i / v
                else:
                    new_ub_i = ub_i if v > 0 else -ub_i
                lb_j = dst.variables.get_lower_bounds(var_j)
                ub_j = dst.variables.get_upper_bounds(var_j)
                if v > 0:
                    dst.variables.set_lower_bounds(var_j, max(lb_j, new_lb_i))
                    dst.variables.set_upper_bounds(var_j, min(ub_j, new_ub_i))
                else:
                    dst.variables.set_lower_bounds(var_j, max(lb_j, new_ub_i))
                    dst.variables.set_upper_bounds(var_j, min(ub_j, new_lb_i))

        for var in dst.variables.get_names():
            lowerbound = dst.variables.get_lower_bounds(var)
            upperbound = dst.variables.get_upper_bounds(var)
            if lowerbound > upperbound:
                logger.warning(
                    'Infeasible lower and upper bound: %s %f %f', var, lowerbound, upperbound)
                return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

    def _objective(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst
        subs = self._subs

        # initialize
        offset = [src.objective.get_offset()]
        lin_dict = {}
        quad_dict = {}

        # substitute quadratic terms of the objective function
        for (i, j), w_ij in src.objective.get_quadratic_dict().items():
            repl_i = subs[i] if i in subs else (i, 1)
            repl_j = subs[j] if j in subs else (j, 1)
            idx = tuple(x for x, _ in [repl_i, repl_j] if x != self.CONST)
            prod = w_ij * repl_i[1] * repl_j[1]
            if len(idx) == 2:
                if idx not in quad_dict:
                    quad_dict[idx] = 0
                quad_dict[idx] += prod
            elif len(idx) == 1:
                k = idx[0]
                if k not in lin_dict:
                    lin_dict[k] = 0
                lin_dict[k] += prod / 2
            else:
                offset.append(prod / 2)

        # substitute linear terms of the objective function
        for i, w_i in src.objective.get_linear_dict().items():
            repl_i = subs[i] if i in subs else (i, 1)
            prod = w_i * repl_i[1]
            if repl_i[0] == self.CONST:
                offset.append(prod)
            else:
                k = repl_i[0]
                if k not in lin_dict:
                    lin_dict[k] = 0
                lin_dict[k] += prod

        dst.objective.set_offset(fsum(offset))
        if len(lin_dict) > 0:
            ind, val = self._replace_dict_keys_with_names(src, lin_dict)
            dst.objective.set_linear([(i, v) for i, v in zip(ind, val) if v != 0])
        if len(quad_dict) > 0:
            ind_pair, val = self._replace_dict_keys_with_names(src, quad_dict)
            ind1, ind2 = zip(*ind_pair)
            dst.objective.set_quadratic_coefficients(
                [(i, j, v) for i, j, v in zip(ind1, ind2, val) if v != 0])

        return SubstitutionStatus.success

    def _linear_constraints(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst
        subs = self._subs

        for name, row, rhs, sense, range_value in zip(
                src.linear_constraints.get_names(),
                src.linear_constraints.get_rows(),
                src.linear_constraints.get_rhs(),
                src.linear_constraints.get_senses(),
                src.linear_constraints.get_range_values()
        ):
            lin_dict = {}
            rhs = [rhs]
            for i, w_i in zip(row.ind, row.val):
                repl_i = subs[i] if i in subs else (i, 1)
                prod = w_i * repl_i[1]
                if repl_i[0] == self.CONST:
                    rhs.append(-prod)
                else:
                    k = repl_i[0]
                    if k not in lin_dict:
                        lin_dict[k] = 0
                    lin_dict[k] += prod
            if len(lin_dict) > 0:
                ind, val = self._replace_dict_keys_with_names(src, lin_dict)
                dst.linear_constraints.add(
                    lin_expr=[SparsePair(ind=ind, val=val)],
                    senses=sense, rhs=[fsum(rhs)], range_values=[range_value], names=[name])
            else:
                if not self._feasible(sense, fsum(rhs), range_value):
                    logger.warning('constraint %s is infeasible due to substitution', name)
                    return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

    def _quadratic_constraints(self) -> SubstitutionStatus:
        src = self._src
        dst = self._dst
        subs = self._subs

        for name, lin_expr, quad_expr, sense, rhs in zip(
                src.quadratic_constraints.get_names(),
                src.quadratic_constraints.get_linear_components(),
                src.quadratic_constraints.get_quadratic_components(),
                src.quadratic_constraints.get_senses(),
                src.quadratic_constraints.get_rhs()
        ):
            quad_dict = {}
            lin_dict = {}
            rhs = [rhs]
            for i, j, w_ij in zip(quad_expr.ind1, quad_expr.ind2, quad_expr.val):
                repl_i = subs[i] if i in subs else (i, 1)
                repl_j = subs[j] if j in subs else (j, 1)
                idx = tuple(x for x, _ in [repl_i, repl_j] if x != self.CONST)
                prod = w_ij * repl_i[1] * repl_j[1]
                if len(idx) == 2:
                    if idx[0] < idx[1]:
                        idx = (idx[1], idx[0])
                    if idx not in quad_dict:
                        quad_dict[idx] = 0
                    quad_dict[idx] += prod
                elif len(idx) == 1:
                    k = idx[0]
                    if k not in lin_dict:
                        lin_dict[k] = 0
                    lin_dict[k] += prod
                else:
                    rhs.append(-prod)
            for i, w_i in zip(lin_expr.ind, lin_expr.val):
                repl_i = subs[i] if i in subs else (i, 1)
                prod = w_i * repl_i[1]
                if repl_i[0] == self.CONST:
                    rhs.append(-prod)
                else:
                    k = repl_i[0]
                    if k not in lin_dict:
                        lin_dict[k] = 0
                    lin_dict[k] += prod
            ind, val = self._replace_dict_keys_with_names(src, lin_dict)
            lin_expr = SparsePair(ind=ind, val=val)
            if len(quad_dict) > 0:
                ind_pair, val = self._replace_dict_keys_with_names(src, quad_dict)
                ind1, ind2 = zip(*ind_pair)
                quad_expr = SparseTriple(ind1=ind1, ind2=ind2, val=val)
                dst.quadratic_constraints.add(
                    name=name,
                    lin_expr=lin_expr,
                    quad_expr=quad_expr,
                    sense=sense,
                    rhs=fsum(rhs)
                )
            elif len(lin_dict) > 0:
                lin_names = set(dst.linear_constraints.get_names())
                while name in lin_names:
                    name = '_' + name
                dst.linear_constraints.add(
                    names=[name],
                    lin_expr=[lin_expr],
                    senses=sense,
                    rhs=[fsum(rhs)]
                )
            else:
                if not self._feasible(sense, fsum(rhs)):
                    logger.warning('constraint %s is infeasible due to substitution', name)
                    return SubstitutionStatus.infeasible

        return SubstitutionStatus.success

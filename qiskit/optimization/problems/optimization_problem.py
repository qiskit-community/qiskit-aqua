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

"""The optimization problem."""

from docplex.mp.model import Model

from qiskit.optimization.problems.linear_constraint import LinearConstraintInterface
from qiskit.optimization.problems.objective import ObjectiveInterface
from qiskit.optimization.problems.problem_type import ProblemType
from qiskit.optimization.problems.quadratic_constraint import QuadraticConstraintInterface
from qiskit.optimization.problems.variables import VariablesInterface
from qiskit.optimization.results.solution import SolutionInterface
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError


class OptimizationProblem:
    """A class encapsulating an optimization problem, modeled after Python CPLEX API.

    An instance of the OptimizationProblem class provides methods for creating,
    modifying, and querying an optimization problem, solving it, and
    querying aspects of the solution.
    """

    def __init__(self, *args):
        """Constructor of the OptimizationProblem class.

        The OptimizationProblem constructor accepts four types of argument lists.

        op = qiskit.optimization.OptimizationProblem()
        op is a new problem with no data

        op = qiskit.optimization.OptimizationProblem("filename")
        op is a new problem containing the data in filename.  If
        filename does not exist, an exception is raised.

        The OptimizationProblem object is a context manager and can be used, like so:

        with qiskit.optimization.OptimizationProblem() as op:
            # do stuff
            op.solve()

        When the with block is finished, the end() method will be called
        automatically.
        """
        from cplex.exceptions import CplexSolverError
        if len(args) > 1:
            raise QiskitOptimizationError("Too many arguments to OptimizationProblem()")
        self._disposed = False
        self._name = None

        # see `qiskit.optimization.VariablesInterface()`
        self.variables = VariablesInterface()

        # see `qiskit.optimization.LinearConstraintInterface()`
        self.linear_constraints = LinearConstraintInterface(varindex=self.variables.get_indices)

        # see `qiskit.optimization.QuadraticConstraintInterface()`
        self.quadratic_constraints = QuadraticConstraintInterface(
            varindex=self.variables.get_indices
        )

        # see `qiskit.optimization.ObjectiveInterface()`
        # pylint: disable=unexpected-keyword-arg
        self.objective = ObjectiveInterface(varindex=self.variables.get_indices)

        # see `qiskit.optimization.SolutionInterface()`
        self.solution = SolutionInterface()

        # see `qiskit.optimization.ProblemType()` -- essentially conversions from integers to
        # strings and back
        self.problem_type = ProblemType()
        self.my_problem_type = 0

        # read from file in case filename is given
        if len(args) == 1:
            try:
                self.read(args[0])
            except CplexSolverError:
                raise QiskitOptimizationError('Could not load file: %s' % args[0])

    def from_cplex(self, op):
        """ from cplex """
        # make sure current problem is clean
        from cplex.exceptions import CplexSolverError
        self._disposed = False
        try:
            self._name = op.get_problem_name()
        except CplexSolverError:
            self._name = None
        self.variables = VariablesInterface()
        self.linear_constraints = LinearConstraintInterface(varindex=self.variables.get_indices)
        self.quadratic_constraints = QuadraticConstraintInterface(
            varindex=self.variables.get_indices)
        self.objective = ObjectiveInterface(varindex=self.variables.get_indices)
        self.solution = SolutionInterface()

        # set problem name
        if op.get_problem_name():
            self.set_problem_name(op.get_problem_name())

        # TODO: how to choose problem type?
        # set problem type
        if op.get_problem_type():
            self.set_problem_type(op.get_problem_type())

        # TODO: There seems to be a bug in CPLEX, it raises a "Not a MIP (3003)"-error
        # if the problem never had a non-cts. variable
        idx = op.variables.add(types='B')
        op.variables.delete(idx[0])

        # set variables (obj is set via objective interface)
        var_names = op.variables.get_names()
        var_lbs = op.variables.get_lower_bounds()
        var_ubs = op.variables.get_upper_bounds()
        var_types = op.variables.get_types()
        self.variables.add(lb=var_lbs, ub=var_ubs, types=var_types, names=var_names)

        # set objective sense
        self.objective.set_sense(op.objective.get_sense())

        # set objective name
        try:
            self.objective.set_name(op.objective.get_name())
        except CplexSolverError:
            pass

        # set linear objective terms
        for i, v in enumerate(op.objective.get_linear()):
            self.objective.set_linear(i, v)

        # set quadratic objective terms
        for i, sparse_pair in enumerate(op.objective.get_quadratic()):
            for j, v in zip(sparse_pair.ind, sparse_pair.val):
                self.objective.set_quadratic_coefficients(i, j, v)

        # set objective offset
        self.objective.set_offset(op.objective.get_offset())

        # set linear constraints
        linear_rows = op.linear_constraints.get_rows()
        linear_sense = op.linear_constraints.get_senses()
        linear_rhs = op.linear_constraints.get_rhs()
        linear_ranges = op.linear_constraints.get_range_values()
        linear_names = op.linear_constraints.get_names()
        self.linear_constraints.add(linear_rows, linear_sense,
                                    linear_rhs, linear_ranges, linear_names)

        # TODO: add quadratic constraints

    def from_docplex(self, model: Model):
        """ from docplex """
        from cplex.exceptions import CplexSolverError
        cplex = model.get_cplex()
        try:
            cplex.set_problem_name(model.get_name())
        except CplexSolverError:
            cplex.set_problem_name('')
        cplex.objective.set_name('Objective')
        self.from_cplex(cplex)

    def to_cplex(self):
        """ to cplex """
        from cplex import Cplex
        # create empty CPLEX model
        op = Cplex()
        if self.get_problem_name() is not None:
            op.set_problem_name(self.get_problem_name())
        else:
            op.set_problem_name('')
        # TODO: what about problem type?

        # set variables (obj is set via objective interface)
        var_names = self.variables.get_names()
        var_lbs = self.variables.get_lower_bounds()
        var_ubs = self.variables.get_upper_bounds()
        var_types = self.variables.get_types()
        # TODO: what about columns?
        op.variables.add(lb=var_lbs, ub=var_ubs, types=var_types, names=var_names)

        # set objective sense
        op.objective.set_sense(self.objective.get_sense())

        # set objective name
        op.objective.set_name(self.objective.get_name())

        # set linear objective terms
        for i, v in self.objective.get_linear().items():
            op.objective.set_linear(i, v)

        # set quadratic objective terms
        for i, v_i in self.objective.get_quadratic().items():
            for j, v in v_i.items():
                op.objective.set_quadratic_coefficients(i, j, v)

        # set objective offset
        op.objective.set_offset(self.objective.get_offset())

        # set linear constraints
        linear_rows = self.linear_constraints.get_rows()
        linear_sense = self.linear_constraints.get_senses()
        linear_rhs = self.linear_constraints.get_rhs()
        linear_ranges = self.linear_constraints.get_range_values()
        linear_names = self.linear_constraints.get_names()
        op.linear_constraints.add(linear_rows, linear_sense, linear_rhs,
                                  linear_ranges, linear_names)

        # TODO: add quadratic constraints

        return op

    def end(self):
        """Releases the OptimizationProblem object."""
        if self._disposed:
            return
        self._disposed = True

    def __del__(self):
        """non-public"""
        self.end()

    def __enter__(self):
        """To implement a ContextManager, as in Cplex."""
        return self

    def __exit__(self, *exc):
        """To implement a ContextManager, as in Cplex."""
        return False

    def read(self, filename, filetype=""):
        """Reads a problem from file.

        The first argument is a string specifying the filename from
        which the problem will be read.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.read("lpex.mps")
        """
        from cplex import Cplex
        cplex = Cplex()
        cplex.read(filename, filetype)
        self.from_cplex(cplex)

    def write(self, filename, filetype=""):
        """Writes a problem to file.

        The first argument is a string specifying the filename to
        which the problem will be written.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names=['x1', 'x2', 'x3'])
        >>> op.write("example.lp")
        """
        cplex = self.to_cplex()
        cplex.write(filename, filetype)

    def write_to_stream(self, stream, filetype='LP', comptype=''):
        """Writes a problem to a file-like object in the given file format.

        The filetype argument can be any of "sav" (a binary format), "lp"
        (the default), "mps", "rew", "rlp", or "alp" (see `OptimizationProblem.write`
        for an explanation of these).

        If comptype is "bz2" (for BZip2) or "gz" (for GNU Zip), a
        compressed file is written.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
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
        try:
            callable(stream.write)
        except AttributeError:
            raise QiskitOptimizationError("stream must have a write method")
        try:
            callable(stream.flush)
        except AttributeError:
            raise QiskitOptimizationError("stream must have a flush method")
        op = self.to_cplex()
        return op.write_to_stream(stream, filetype, comptype)

    def write_as_string(self, filetype='LP', comptype=''):
        """Writes a problem as a string in the given file format.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names=['x1', 'x2', 'x3'])
        >>> lp_str = op.write_as_string("lp")
        >>> len(lp_str) > 0
        True
        """
        op = self.to_cplex()
        return op.write_as_string(filetype, comptype)

    def get_problem_type(self):
        """Returns the problem type.

        The return value is an attribute of self.problem_type.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.read("lpex.mps")
        >>> op.get_problem_type()
        0
        >>> op.problem_type[op.get_problem_type()]
        'LP'
        """
        # TODO: A better option would be to scan the variables to check their types, etc.
        return self.my_problem_type

    def set_problem_type(self, _type):
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
        self.my_problem_type = _type

    def solve(self):
        """Solves the problem.

        Note
          The solve method returning normally does not necessarily mean
          that an optimal or feasible solution has been found.  Use
          OptimizationProblem.solution.get_status() to query the status of the current
          solution.
        """
        # TODO: Implement me
        pass

    def set_problem_name(self, name):
        """Set the problem name."""
        self._name = name

    def get_problem_name(self):
        """Get the problem name."""
        return self._name

    def substitute_variables(self, constants=None, variables=None):
        """Substitute variables of the problem.

        constants: SparsePair (replace variable by constant)
        variables: SparseTriple (replace variables by weighted other variable
        need to copy everything using name reference to make sure that indices are matched correctly
        """
        from cplex import SparsePair
        # guarantee that there is no overlap between variables to be replaced and combine input
        vars_to_be_replaced = {}
        if constants is not None:
            for i, v in zip(constants.ind, constants.val):
                i = self.variables.get_indices(i)
                name = self.variables.get_names(i)
                if i in vars_to_be_replaced:
                    raise QiskitOptimizationError('cannot substitute the same variable twice')
                vars_to_be_replaced[name] = [v]
        if variables is not None:
            for i, j, v in zip(variables.ind1, variables.ind2, variables.val):
                i = self.variables.get_indices(i)
                j = self.variables.get_indices(j)
                name1 = self.variables.get_names(i)
                name2 = self.variables.get_names(j)
                if name1 in vars_to_be_replaced:
                    raise QiskitOptimizationError('Cannot substitute the same variable twice')
                if name2 in vars_to_be_replaced.keys():
                    raise QiskitOptimizationError(
                        'Cannot substitute by variable that gets substituted it self.')
                vars_to_be_replaced[name1] = [name2, v]

        # get variables to be kept
        vars_to_be_kept = set()
        for name in self.variables.get_names():
            if name not in vars_to_be_replaced:
                vars_to_be_kept.add(name)

        # construct new problem
        op = OptimizationProblem()

        # set problem name
        op.set_problem_name(self.get_problem_name())

        # copy variables that are not replaced
        # TODO: what about columns?
        for name, var_type, lower_bound, upper_bound in zip(
                self.variables.get_names(),
                self.variables.get_types(),
                self.variables.get_lower_bounds(),
                self.variables.get_upper_bounds(),
        ):
            if name not in vars_to_be_replaced:
                op.variables.add(lb=[lower_bound], ub=[upper_bound], types=[var_type], names=[name])
            else:
                # check that replacement satisfies bounds
                repl = vars_to_be_replaced[name]
                if len(repl) == 1:
                    if not lower_bound <= repl[0] <= upper_bound:
                        raise QiskitOptimizationError('Infeasible substitution for variable')

        # initialize offset
        offset = self.objective.get_offset()

        # construct linear part of objective
        for i, v in self.objective.get_linear().items():
            i = self.variables.get_indices(i)
            i_name = self.variables.get_names(i)
            i_repl = vars_to_be_replaced.get(i_name, None)
            if i_repl is not None:
                w_i = self.objective.get_linear(i_name)
                if len(i_repl) == 1:
                    offset += i_repl[0] * w_i
                else:  # len == 2
                    w_i = i_repl[1] * w_i + op.objective.get_linear(i_repl[0])
                    op.objective.set_linear(i_repl[0], w_i)
            else:
                w_i = self.objective.get_linear(i_name) + op.objective.get_linear(i_name)
                op.objective.set_linear(i_name, w_i)

        # construct quadratic part of objective
        for i, v_i in self.objective.get_quadratic().items():
            for j, v in v_i.items():
                i = self.variables.get_indices(i)
                j = self.variables.get_indices(j)
                i_name = self.variables.get_names(i)
                j_name = self.variables.get_names(j)
                i_repl = vars_to_be_replaced.get(i_name, None)
                j_repl = vars_to_be_replaced.get(j_name, None)
                w_ij = self.objective.get_quadratic_coefficients(i_name, j_name)
                if i_repl is not None and j_repl is None:
                    if len(i_repl) == 1:
                        # if x_i is replaced, the term needs to be added to the linear part of x_j
                        w_j = op.objective.get_linear(j_name)
                        w_j += i_repl[0] * w_ij / 2
                        op.objective.set_linear(j_name, w_j)
                    else:  # len == 2
                        k = self.variables.get_indices(i_repl[0])
                        k_name = self.variables.get_names(k)
                        if k_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        w_jk = op.objective.get_quadratic_coefficients(j_name, k_name)
                        w_jk += i_repl[1] * w_ij
                        op.objective.set_quadratic_coefficients(j_name, k_name, w_jk)
                elif i_repl is None and j_repl is not None:
                    if len(j_repl) == 1:
                        # if x_j is replaced, the term needs to be added to the linear part of x_i
                        w_i = op.objective.get_linear(i_name)
                        w_i += j_repl[0] * w_ij / 2
                        op.objective.set_linear(i_name, w_i)
                    else:  # len == 2
                        k = self.variables.get_indices(j_repl[0])
                        k_name = self.variables.get_names(k)
                        if k_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        w_ik = op.objective.get_quadratic_coefficients(i_name, k_name)
                        w_ik += j_repl[1] * w_ij
                        op.objective.set_quadratic_coefficients(i_name, k_name, w_ik)
                elif i_repl is not None and j_repl is not None:
                    if len(i_repl) == 1 and len(j_repl) == 1:
                        offset += w_ij * i_repl[0] * j_repl[0] / 2
                    elif len(i_repl) == 1 and len(j_repl) == 2:
                        k = self.variables.get_indices(j_repl[0])
                        k_name = self.variables.get_names(k)
                        if k_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        w_k = op.objective.get_linear(k_name)
                        w_k += w_ij * i_repl[0] * j_repl[1] / 2
                        op.objective.set_linear(k_name, w_k)
                    elif len(i_repl) == 2 and len(j_repl) == 1:
                        k = self.variables.get_indices(i_repl[0])
                        k_name = self.variables.get_names(k)
                        if k_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        w_k = op.objective.get_linear(k_name)
                        w_k += w_ij * j_repl[0] * i_repl[1] / 2
                        op.objective.set_linear(k_name, w_k)
                    else:  # both len(repl) == 2
                        k = self.variables.get_indices(i_repl[0])
                        k_name = self.variables.get_names(k)
                        if k_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        m = self.variables.get_indices(j_repl[0])
                        m_name = self.variables.get_names(m)
                        if m_name in vars_to_be_replaced.keys():
                            raise QiskitOptimizationError(
                                'Cannot substitute by variable that gets substituted itself.')
                        w_kl = op.objective.get_quadratic_coefficients(k_name, m_name)
                        w_kl += w_ij * i_repl[1] * j_repl[1]
                        op.objective.set_quadratic_coefficients(k_name, m_name, w_kl)
                else:
                    # nothing to be replaced, just copy coefficients
                    if i == j:
                        w_ij = sum([self.objective.get_quadratic_coefficients(i_name, j_name),
                                    op.objective.get_quadratic_coefficients(i_name, j_name)])
                    else:
                        w_ij = sum([self.objective.get_quadratic_coefficients(i_name, j_name) / 2,
                                    op.objective.get_quadratic_coefficients(i_name, j_name)])
                    op.objective.set_quadratic_coefficients(i_name, j_name, w_ij)

        # set offset
        op.objective.set_offset(offset)

        # construct linear constraints
        for name, row, rhs, sense, range_value in zip(
                self.linear_constraints.get_names(),
                self.linear_constraints.get_rows(),
                self.linear_constraints.get_rhs(),
                self.linear_constraints.get_senses(),
                self.linear_constraints.get_range_values()
        ):
            # print(name, row, rhs, sense, range_value)
            new_vals = {}
            for i, v in zip(row.ind, row.val):
                i = self.variables.get_indices(i)
                i_name = self.variables.get_names(i)
                i_repl = vars_to_be_replaced.get(i_name, None)
                if i_repl is not None:
                    if len(i_repl) == 1:
                        rhs -= v * i_repl[0]
                    else:
                        j = self.variables.get_indices(i_repl[0])
                        j_name = self.variables.get_names(j)
                        new_vals[j_name] = v * i_repl[1] + new_vals.get(i_name, 0)
                else:
                    # nothing to replace, just add value
                    new_vals[i_name] = v + new_vals.get(i_name, 0)
            new_ind = list(new_vals.keys())
            new_val = [new_vals[i] for i in new_ind]
            new_row = SparsePair(new_ind, new_val)
            op.linear_constraints.add(
                lin_expr=[new_row], senses=[sense], rhs=[rhs], range_values=[range_value],
                names=[name])

        # TODO: quadratic constraints

        # TODO: amend self.my_problem_type

        return op

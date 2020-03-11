# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Sequence

import copy

from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError
from qiskit.optimization.utils.helpers import init_list_args, validate_arg_lengths, listify, convert
from cplex import SparsePair


# TODO: can we delete these?
CPX_CON_LOWER_BOUND = 1
CPX_CON_UPPER_BOUND = 2
CPX_CON_LINEAR = 3
CPX_CON_QUADRATIC = 4
CPX_CON_SOS = 5
CPX_CON_INDICATOR = 6
CPX_CON_PWL = 7
CPX_CON_ABS = 7
CPX_CON_MINEXPR = 8
CPX_CON_MAXEXPR = 9
CPX_CON_LAST_CONTYPE = 10


class LinearConstraintInterface(BaseInterface):
    """Methods for adding, modifying, and querying linear constraints."""

    def __init__(self, varsgetindexfunc=None):
        """Creates a new LinearConstraintInterface.

        The linear constraints interface is exposed by the top-level
        `OptimizationProblem` class as `OptimizationProblem.linear_constraints`.
        This constructor is not meant to be used externally.
        """
        super(LinearConstraintInterface, self).__init__()
        self._rhs = []
        self._senses = []
        self._range_values = []
        self._names = []
        self._lin_expr = []
        self._linconsgetindex = {}
        self._varsgetindexfunc = varsgetindexfunc

    def get_num(self):
        """Returns the number of linear constraints.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c1", "c2", "c3"])
        >>> op.linear_constraints.get_num()
        3
        """
        return len(self._names)

    def _linconsgetindexfunc(self, item):
        if item not in self._linconsgetindex:
            self._linconsgetindex[item] = len(self._linconsgetindex)
        return self._linconsgetindex[item]

    def _linconsrebuildindex(self):
        self._linconsgetindex = {}
        for (cnt, item) in enumerate(self._names):
            self._linconsgetindex[item] = cnt

    def add(self, lin_expr=None, senses="", rhs=None, range_values=None,
            names=None):
        """Adds linear constraints to the problem.

        linear_constraints.add accepts the keyword arguments lin_expr,
        senses, rhs, range_values, and names.

        If more than one argument is specified, all arguments must
        have the same length.

        lin_expr may be either a list of SparsePair instances or a
        matrix in list-of-lists format.

        Note
          The entries of lin_expr must not contain duplicate indices.
          If an entry of lin_expr references a variable more than
          once, either by index, name, or a combination of index and
          name, an exception will be raised.

        senses must be either a list of single-character strings or a
        string containing the senses of the linear constraints.
        Each entry must
        be one of 'G', 'L', 'E', and 'R', indicating greater-than,
        less-than, equality, and ranged constraints, respectively.

        rhs is a list of floats, specifying the righthand side of
        each linear constraint.

        range_values is a list of floats, specifying the difference
        between lefthand side and righthand side of each linear constraint.
        If range_values[i] > 0 (zero) then the constraint i is defined as
        rhs[i] <= rhs[i] + range_values[i]. If range_values[i] < 0 (zero)
        then constraint i is defined as
        rhs[i] + range_value[i] <= a*x <= rhs[i].

        names is a list of strings.

        Returns an iterator containing the indices of the added linear
        constraints.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ["x1", "x2", "x3"])
        >>> indices = op.linear_constraints.add(\
                lin_expr = [SparsePair(ind = ["x1", "x3"], val = [1.0, -1.0]),\
                            SparsePair(ind = ["x1", "x2"], val = [1.0, 1.0]),\
                            SparsePair(ind = ["x1", "x2", "x3"], val = [-1.0] * 3),\
                            SparsePair(ind = ["x2", "x3"], val = [10.0, -2.0])],\
                senses = ["E", "L", "G", "R"],\
                rhs = [0.0, 1.0, -1.0, 2.0],\
                range_values = [0.0, 0.0, 0.0, -10.0],\
                names = ["c0", "c1", "c2", "c3"])
        >>> op.linear_constraints.get_rhs()
        [0.0, 1.0, -1.0, 2.0]
        """

        arg_list = init_list_args(lin_expr, senses, rhs, range_values, names)
        arg_lengths = [len(x) for x in arg_list]
        if len(arg_lengths) == 0:
            return range(0)
        max_length = max(arg_lengths)
        for arg_length in arg_lengths:
            if arg_length > 0 and arg_length != max_length:
                raise QiskitOptimizationError("inconsistent arguments in linear_constraints.add().")

        if max_length > 0:

            if not rhs:
                rhs = [0.0] * max_length
            self._rhs.extend(rhs)

            if not senses:
                senses = "E" * max_length
            self._senses.extend(senses)

            if not range_values:
                range_values = [0.0] * max_length
            self._range_values.extend(range_values)

            if not names:
                names = ["c" + str(cnt) for cnt in range(len(self._names),
                                                         len(self._names) + max_length)]
            self._names.extend(names)
            for name in names:
                self._linconsgetindexfunc(name)

            if not lin_expr:
                lin_expr = [SparsePair()] * max_length
            if all(isinstance(el, SparsePair) for el in lin_expr):
                for sp in lin_expr:
                    lin_expr_dict = {}
                    for i, val in zip(sp.ind, sp.val):
                        i = convert(i, self._varsgetindexfunc)
                        if i in lin_expr_dict:
                            raise QiskitOptimizationError(
                                'Variables should only appear once in linear constraint.')
                        lin_expr_dict[i] = val
                    self._lin_expr += [lin_expr_dict]
            elif all(isinstance(el, Sequence) for el in lin_expr):
                for l in lin_expr:
                    lin_expr_dict = {}
                    for i, val in zip(l[0], l[1]):
                        i = convert(i, self._varsgetindexfunc)
                        if i in lin_expr_dict:
                            raise QiskitOptimizationError(
                                'Variables should only appear once in linear constraint.')
                        lin_expr_dict[i] = val
                    self._lin_expr += [lin_expr_dict]
            else:
                raise QiskitOptimizationError(
                    'Invalid lin_expr format in linear_constraint.add().')

        return range(len(self._names) - max_length, len(self._names))

    def delete(self, *args):
        """Removes linear constraints from the problem.

        There are four forms by which linear_constraints.delete may be
        called.

        linear_constraints.delete()
          deletes all linear constraints from the problem.

        linear_constraints.delete(i)
          i must be a linear constraint name or index. Deletes the
          linear constraint whose index or name is i.

        linear_constraints.delete(s)
          s must be a sequence of linear constraint names or indices.
          Deletes the linear constraints with names or indices contained
          within s. Equivalent to [linear_constraints.delete(i) for i in s].

        linear_constraints.delete(begin, end)
          begin and end must be linear constraint indices or linear
          constraint names. Deletes the linear constraints with indices
          between begin and end, inclusive of end. Equivalent to
          linear_constraints.delete(range(begin, end + 1)). This will
          give the best performance when deleting batches of linear
          constraints.

        See CPXdelrows in the Callable Library Reference Manual for
        more detail.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names=[str(i) for i in range(10)])
        >>> op.linear_constraints.get_num()
        10
        >>> op.linear_constraints.delete(8)
        >>> op.linear_constraints.get_names()
        ['0', '1', '2', '3', '4', '5', '6', '7', '9']
        >>> op.linear_constraints.delete("1", 3)
        >>> op.linear_constraints.get_names()
        ['0', '4', '5', '6', '7', '9']
        >>> op.linear_constraints.delete([2, "0", 5])
        >>> op.linear_constraints.get_names()
        ['4', '6', '7']
        >>> op.linear_constraints.delete()
        >>> op.linear_constraints.get_names()
        []
        """

        # TODO: delete does not update the index to find constraints by name etc.

        def _delete(i):
            del self._rhs[i]
            del self._senses[i]
            del self._names[i]
            del self._lin_expr[i]
            del self._range_values[i]

        if len(args) == 0:
            # Delete All:
            self._rhs = []
            self._senses = []
            self._names = []
            self._lin_expr = []
            self._range_values = []
            self._linconsgetindex = {}
        elif len(args) == 1:
            # Delete all items from a possibly unordered list of mixed types:
            args = listify(convert(args[0], self._linconsgetindexfunc))
            args = sorted(args)
            for i, j in enumerate(args):
                _delete(j - i)
            self._linconsrebuildindex()
        elif len(args) == 2:
            # Delete range from arg[0] to arg[1]:
            start = convert(args[0], self._linconsgetindexfunc)
            end = convert(args[1], self._linconsgetindexfunc)
            self.delete(range(start, end + 1))
            self._linconsrebuildindex()
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_rhs(self, *args):
        """Sets the righthand side of a set of linear constraints.

        There are two forms by which linear_constraints.set_rhs may be
        called.

        linear_constraints.set_rhs(i, rhs)
          i must be a row name or index and rhs must be a real number.
          Sets the righthand side of the row whose index or name is
          i to rhs.

        linear_constraints.set_rhs(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, rhs) pairs, each
          of which consists of a row name or index and a real
          number.  Sets the righthand side of the specified rows to
          the corresponding values.  Equivalent to
          [linear_constraints.set_rhs(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> op.linear_constraints.get_rhs()
        [0.0, 0.0, 0.0, 0.0]
        >>> op.linear_constraints.set_rhs("c1", 1.0)
        >>> op.linear_constraints.get_rhs()
        [0.0, 1.0, 0.0, 0.0]
        >>> op.linear_constraints.set_rhs([("c3", 2.0), (2, -1.0)])
        >>> op.linear_constraints.get_rhs()
        [0.0, 1.0, -1.0, 2.0]
        """

        def _set(i, v):
            self._rhs[convert(i, self._linconsgetindexfunc)] = v

        if len(args) == 2:
            _set(args[0], args[1])
        elif len(args) == 1:
            args = listify(args[0])
            for (i, v) in args:
                _set(i, v)
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_names(self, *args):
        """Sets the name of a linear constraint or set of linear constraints.

        There are two forms by which linear_constraints.set_names may be
        called.

        linear_constraints.set_names(i, name)
          i must be a linear constraint name or index and name must be a
          string.

        linear_constraints.set_names(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, name) pairs,
          each of which consists of a linear constraint name or index and a
          string.  Sets the name of the specified linear constraints to the
          corresponding strings.  Equivalent to
          [linear_constraints.set_names(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> op.linear_constraints.set_names("c1", "second")
        >>> op.linear_constraints.get_names(1)
        'second'
        >>> op.linear_constraints.set_names([("c3", "last"), (2, "middle")])
        >>> op.linear_constraints.get_names()
        ['c0', 'second', 'middle', 'last']
        """

        def _set(i, v):
            self._names[convert(i, self._linconsgetindexfunc)] = v

        if len(args) == 2:
            _set(args[0], args[1])
        elif len(args) == 1:
            args = listify(args[0])
            for (i, v) in args:
                _set(i, v)
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_senses(self, *args):
        """Sets the sense of a linear constraint or set of linear constraints.

        There are two forms by which linear_constraints.set_senses may be
        called.

        linear_constraints.set_senses(i, type)
          i must be a row name or index and name must be a
          single-character string.

        linear_constraints.set_senses(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, sense) pairs,
          each of which consists of a row name or index and a
          single-character string.  Sets the sense of the specified
          rows to the corresponding strings.  Equivalent to
          [linear_constraints.set_senses(pair[0], pair[1]) for pair in seq_of_pairs].

        The senses of the constraints must be one of 'G', 'L', 'E',
        and 'R', indicating greater-than, less-than, equality, and
        ranged constraints, respectively.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> op.linear_constraints.get_senses()
        ['E', 'E', 'E', 'E']
        >>> op.linear_constraints.set_senses("c1", "G")
        >>> op.linear_constraints.get_senses(1)
        'G'
        >>> op.linear_constraints.set_senses([("c3", "L"), (2, "R")])
        >>> op.linear_constraints.get_senses()
        ['E', 'G', 'R', 'L']
        """

        def _set(i, v):
            v = v.upper().strip()
            if v in ["G", "L", "E", "R"]:
                self._senses[convert(i, getindexfunc=self._linconsgetindexfunc)] = v
            else:
                raise QiskitOptimizationError("Wrong sense!")

        if len(args) == 2:
            _set(args[0], args[1])
        elif len(args) == 1:
            args = listify(args[0])
            for (i, v) in args:
                _set(i, v)
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_linear_components(self, *args):
        """Sets a linear constraint or set of linear constraints.

        There are two forms by which this method may be called:

        linear_constraints.set_linear_components(i, lin)
          i must be a row name or index and lin must be either a
          SparsePair or a pair of sequences, the first of which
          consists of variable names or indices, the second of which
          consists of floats.

        linear_constraints.set_linear_components(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, lin) pairs,
          each of which consists of a row name or index and a vector
          as described above.  Sets the specified rows
          to the corresponding vector.  Equivalent to
          [linear_constraints.set_linear_components(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> indices = op.variables.add(names = ["x0", "x1"])
        >>> op.linear_constraints.set_linear_components("c0", [["x0"], [1.0]])
        >>> op.linear_constraints.get_rows("c0")
        SparsePair(ind = [0], val = [1.0])
        >>> op.linear_constraints.set_linear_components([("c3", SparsePair(ind = ["x1"], val = [-1.0])),\
                                                        (2, [[0, 1], [-2.0, 3.0]])])
        >>> op.linear_constraints.get_rows()
        [SparsePair(ind = [0], val = [1.0]), SparsePair(ind = [], val = []), SparsePair(ind = [0, 1], val = [-2.0, 3.0]), SparsePair(ind = [1], val = [-1.0])]
        """

        def _set(i, v):
            i = convert(i, self._linconsgetindexfunc)
            if isinstance(v, SparsePair):
                ind, val = SparsePair.unpack(v)
                for j, w in zip(ind, val):
                    j = convert(j, self._varsgetindexfunc)
                    self._lin_expr[i][j] = w
            elif isinstance(v, Sequence):
                if len(v) != 2:
                    raise QiskitOptimizationError(
                        "Wrong linear expression. A SparsePair or a pair of indices and values is expected!")
                for j, w in zip(v[0], v[1]):
                    j = convert(j, self._varsgetindexfunc)
                    self._lin_expr[i][j] = w
            else:
                raise QiskitOptimizationError("Wrong linear expression. A SparsePair is expected!")

        if len(args) == 2:
            _set(args[0], args[1])
        elif len(args) == 1:
            args = listify(args[0])
            for (i, v) in args:
                _set(i, v)
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_range_values(self, *args):
        """Sets the range values for a set of linear constraints.

        That is, this method sets the lefthand side (lhs) for each ranged
        constraint of the form lhs <= lin_expr <= rhs.

        The range values are a list of floats, specifying the difference
        between lefthand side and righthand side of each linear constraint.
        If range_values[i] > 0 (zero) then the constraint i is defined as
        rhs[i] <= rhs[i] + range_values[i]. If range_values[i] < 0 (zero)
        then constraint i is defined as
        rhs[i] + range_value[i] <= a*x <= rhs[i].

        Note that changing the range values will not change the sense of a
        constraint; you must call the method set_senses() of the class
        LinearConstraintInterface to change the sense of a ranged row if
        the previous range value was 0 (zero) and the constraint sense was not
        'R'. Similarly, changing the range coefficient from a nonzero value to
        0 (zero) will not change the constraint sense from 'R" to "E"; an
        additional call of setsenses() is required to accomplish that.

        There are two forms by which linear_constraints.set_range_values may be
        called.

        linear_constraints.set_range_values(i, range)
          i must be a row name or index and range must be a real
          number.  Sets the range value of the row whose index or
          name is i to range.

        linear_constraints.set_range_values(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, range) pairs, each
          of which consists of a row name or index and a real
          number.  Sets the range values for the specified rows to
          the corresponding values.  Equivalent to
          [linear_constraints.set_range_values(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> op.linear_constraints.set_range_values("c1", 1.0)
        >>> op.linear_constraints.get_range_values()
        [0.0, 1.0, 0.0, 0.0]
        >>> op.linear_constraints.set_range_values([("c3", 2.0), (2, -1.0)])
        >>> op.linear_constraints.get_range_values()
        [0.0, 1.0, -1.0, 2.0]
        """

        def _set(i, v):
            self._range_values[convert(i, getindexfunc=self._linconsgetindexfunc)] = v
            # TODO: raise QiskitOptimizationError("Wrong range!")

        if len(args) == 2:
            _set(args[0], args[1])
        elif len(args) == 1:
            args = listify(args[0])
            for (i, v) in args:
                _set(i, v)
        else:
            raise QiskitOptimizationError("Wrong number of arguments.")

    def set_coefficients(self, *args):
        """Sets individual coefficients of the linear constraint matrix.

        There are two forms by which
        linear_constraints.set_coefficients may be called.

        linear_constraints.set_coefficients(row, col, val)
          row and col must be indices or names of a linear constraint
          and variable, respectively.  The corresponding coefficient
          is set to val.

        linear_constraints.set_coefficients(coefficients)
          coefficients must be a list of (row, col, val) triples as
          described above.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"])
        >>> indices = op.variables.add(names = ["x0", "x1"])
        >>> op.linear_constraints.set_coefficients("c0", "x1", 1.0)
        >>> op.linear_constraints.get_rows(0)
        SparsePair(ind = [1], val = [1.0])
        >>> op.linear_constraints.set_coefficients([("c2", "x0", 2.0),\
                                                   ("c2", "x1", -1.0)])
        >>> op.linear_constraints.get_rows("c2")
        SparsePair(ind = [0, 1], val = [2.0, -1.0])
        """
        if len(args) == 3:
            arg_list = [args]
        elif len(args) == 1:
            arg_list = listify(args[0])
        else:
            raise QiskitOptimizationError("Wrong number of arguments")
        for ijv in arg_list:
            i = convert(ijv[0], self._linconsgetindexfunc)
            j = convert(ijv[1], self._varsgetindexfunc)
            self._lin_expr[i][j] = ijv[2]

    def get_rhs(self, *args):
        """Returns the righthand side of constraints from the problem.

        Can be called by four forms.

        linear_constraints.get_rhs()
          return the righthand side of all linear constraints from
          the problem.

        linear_constraints.get_rhs(i)
          i must be a linear constraint name or index.  Returns the
          righthand side of the linear constraint whose index or
          name is i.

        linear_constraints.get_rhs(s)
          s must be a sequence of linear constraint names or indices.
          Returns the righthand side of the linear constraints with
          indices the members of s.  Equivalent to
          [linear_constraints.get_rhs(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(rhs = [1.5 * i for i in range(10)],\
                                     names = [str(i) for i in range(10)])
        >>> op.linear_constraints.get_num()
        10
        >>> op.linear_constraints.get_rhs(8)
        12.0
        >>> op.linear_constraints.get_rhs([2,"0",5])
        [3.0, 0.0, 7.5]
        >>> op.linear_constraints.get_rhs()
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """

        def _get(i):
            i = convert(i, self._linconsgetindexfunc)
            return self._rhs[i]

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._rhs)
        elif len(args) == 1:
            if isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            for i in args:
                out.append(_get(i))
        return out

    def get_senses(self, *args):
        """Returns the senses of constraints from the problem.

        Can be called by four forms.

        linear_constraints.get_senses()
          return the senses of all linear constraints from the
          problem.

        linear_constraints.get_senses(i)
          i must be a linear constraint name or index.  Returns the
          sense of the linear constraint whose index or name is i.

        linear_constraints.get_senses(s)
          s must be a sequence of linear constraint names or indices.
          Returns the senses of the linear constraints with indices
          the members of s.  Equivalent to
          [linear_constraints.get_senses(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(
        ...     senses=["E", "G", "L", "R"],
        ...     names=[str(i) for i in range(4)])
        >>> op.linear_constraints.get_num()
        4
        >>> op.linear_constraints.get_senses(1)
        'G'
        >>> op.linear_constraints.get_senses([2,"0",1])
        ['L', 'E', 'G']
        >>> op.linear_constraints.get_senses()
        ['E', 'G', 'L', 'R']
        """

        def _get(i):
            i = convert(i, self._linconsgetindexfunc)
            return self._senses[i]

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._senses)
        elif len(args) == 1:
            if isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            for i in args:
                out.append(_get(i))
        return out

    def get_range_values(self, *args):
        """Returns the range values of linear constraints from the problem.

        That is, this method returns the lefthand side (lhs) for each
        ranged constraint of the form lhs <= lin_expr <= rhs. This method
        makes sense only for ranged constraints, that is, linear constraints
        of sense 'R'.

        The range values are a list of floats, specifying the difference
        between lefthand side and righthand side of each linear constraint.
        If range_values[i] > 0 (zero) then the constraint i is defined as
        rhs[i] <= rhs[i] + range_values[i]. If range_values[i] < 0 (zero)
        then constraint i is defined as
        rhs[i] + range_value[i] <= a*x <= rhs[i].

        Can be called by four forms.

        linear_constraints.get_range_values()
          return the range values of all linear constraints from the
          problem.

        linear_constraints.get_range_values(i)
          i must be a linear constraint name or index.  Returns the
          range value of the linear constraint whose index or name is i.

        linear_constraints.get_range_values(s)
          s must be a sequence of linear constraint names or indices.
          Returns the range values of the linear constraints with
          indices the members of s.  Equivalent to
          [linear_constraints.get_range_values(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(\
                range_values = [1.5 * i for i in range(10)],\
                senses = ["R"] * 10,\
                names = [str(i) for i in range(10)])
        >>> op.linear_constraints.get_num()
        10
        >>> op.linear_constraints.get_range_values(8)
        12.0
        >>> op.linear_constraints.get_range_values([2,"0",5])
        [3.0, 0.0, 7.5]
        >>> op.linear_constraints.get_range_values()
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """

        def _get(i):
            i = convert(i, self._linconsgetindexfunc)
            return self._range_values[i]

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._range_values)
        elif len(args) == 1:
            if isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            for i in args:
                out.append(_get(i))
        return out

    def get_coefficients(self, *args):
        """Returns coefficients by row, column coordinates.

        There are three forms by which
        linear_constraints.get_coefficients may be called.

        Without arguments, it returns a dictionary indexed
        first by constraints and second by variables.

        With two arguments,
        linear_constraints.get_coefficients(row, col)
          returns the coefficient.

        With one argument,
        linear_constraints.get_coefficients(sequence_of_pairs)
          returns a list of coefficients.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ["x0", "x1"])
        >>> indices = op.linear_constraints.add(\
                names = ["c0", "c1"],\
                lin_expr = [[[1], [1.0]], [[0, 1], [2.0, -1.0]]])
        >>> op.linear_constraints.get_coefficients("c0", "x1")
        1.0
        >>> op.linear_constraints.get_coefficients([("c1", "x0"), ("c1", "x1")])
        [2.0, -1.0]
        """

        def _get(i, j):
            i = convert(i, self._linconsgetindexfunc)
            j = convert(j, self._varsgetindexfunc)
            return self._lin_expr[i].get(j, 0)

        if len(args) == 0:
            return copy.deepcopy(self._lin_expr)
        elif len(args) == 1:
            if isinstance(args[0], Sequence):
                out = []
                for (i, j) in args[0]:
                    out.append(_get(i, j))
                return out
            else:
                raise QiskitOptimizationError(
                    "Wrong type of arguments. Single argument must be of list type.")
        elif len(args) == 2:
            return _get(args[0], args[1])
        else:
            raise QiskitOptimizationError(
                "Wrong number of arguments. Please use 2 or one list of pairs.")

    def get_rows(self, *args):
        """Returns a set of rows of the linear constraint matrix.

        Returns a list of SparsePair instances or a single SparsePair
        instance, depending on the form by which it was called.

        There are four forms by which linear_constraints.get_rows may be called.

        linear_constraints.get_rows()
          return the entire linear constraint matrix.

        linear_constraints.get_rows(i)
          i must be a row name or index.  Returns the ith row of
          the linear constraint matrix.

        linear_constraints.get_rows(s)
          s must be a sequence of row names or indices.  Returns the
          rows of the linear constraint matrix indexed by the members
          of s.  Equivalent to
          [linear_constraints.get_rows(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ["x1", "x2", "x3"])
        >>> indices = op.linear_constraints.add(\
                names = ["c0", "c1", "c2", "c3"],\
                lin_expr = [SparsePair(ind = ["x1", "x3"], val = [1.0, -1.0]),\
                            SparsePair(ind = ["x1", "x2"], val = [1.0, 1.0]),\
                            SparsePair(ind = ["x1", "x2", "x3"], val = [-1.0] * 3),\
                            SparsePair(ind = ["x2", "x3"], val = [10.0, -2.0])])
        >>> op.linear_constraints.get_rows(0)
        SparsePair(ind = [0, 2], val = [1.0, -1.0])
        >>> op.linear_constraints.get_rows(["c2", 0])
        [SparsePair(ind = [0, 1, 2], val = [-1.0, -1.0, -1.0]), SparsePair(ind = [0, 2], val = [1.0, -1.0])]
        >>> op.linear_constraints.get_rows()
        [SparsePair(ind = [0, 2], val = [1.0, -1.0]), SparsePair(ind = [0, 1], val = [1.0, 1.0]), SparsePair(ind = [0, 1, 2], val = [-1.0, -1.0, -1.0]), SparsePair(ind = [1, 2], val = [10.0, -2.0])]
        """

        def _get(i):
            i = convert(i, self._linconsgetindexfunc)
            keys = self._lin_expr[i].keys()
            keys = sorted(keys)
            values = [self._lin_expr[i][k] for k in keys]
            s = SparsePair(keys, values)
            return s

        out = []
        if len(args) == 0:
            for i in range(len(self._lin_expr)):
                out.append(_get(i))
            return out
        elif len(args) == 1:
            if isinstance(args[0], str):
                return _get(args[0])
            if isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
                return out
            else:
                return _get(args[0])
        else:
            raise QiskitOptimizationError(
                "Wrong number of arguments. Please use 0 or 1. If there is 1, it can be iterable.")

    def get_num_nonzeros(self):
        """Returns the number of nonzeros in the linear constraint matrix.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ["x1", "x2", "x3"])
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2", "c3"],\
                                     lin_expr = [SparsePair(ind = ["x1", "x3"], val = [1.0, -1.0]),\
                                             SparsePair(ind = ["x1", "x2"], val = [1.0, 1.0]),\
                                             SparsePair(ind = ["x1", "x2", "x3"], val = [-1.0] * 3),\
                                             SparsePair(ind = ["x2", "x3"], val = [10.0, -2.0])])
        >>> op.linear_constraints.get_num_nonzeros()
        9
        """
        nnz = 0
        for c in self._lin_expr:
            for v in c.values():
                nnz += 1 * (v != 0.0)
        return nnz

    def get_names(self, *args):
        """Returns the names of linear constraints from the problem.

        There are four forms by which linear_constraints.get_names may be called.

        linear_constraints.get_names()
          return the names of all linear constraints from the problem.

        linear_constraints.get_names(i)
          i must be a linear constraint index.  Returns the name of row i.

        linear_constraints.get_names(s)
          s must be a sequence of row indices.  Returns the names of
          the linear constraints with indices the members of s.
          Equivalent to [linear_constraints.get_names(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c" + str(i) for i in range(10)])
        >>> op.linear_constraints.get_num()
        10
        >>> op.linear_constraints.get_names(8)
        'c8'
        >>> op.linear_constraints.get_names([2, 0, 5])
        ['c2', 'c0', 'c5']
        >>> op.linear_constraints.get_names()
        ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        """

        def _get(i):
            i = convert(i, self._linconsgetindexfunc)
            return self._names[i]

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._names)
        elif len(args) == 1:
            if isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            for i in args:
                out.append(_get(i))
        return out

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

"""Variable interface"""

import copy
from typing import List, Optional, Union

from qiskit.optimization import infinity
from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.helpers import init_list_args, NameIndex
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError

CPX_CONTINUOUS = 'C'
CPX_BINARY = 'B'
CPX_INTEGER = 'I'
CPX_SEMICONT = 'S'
CPX_SEMIINT = 'N'


class VarTypes:
    """Constants defining variable types

    These constants are compatible with IBM ILOG CPLEX.
    For a definition of each type, see those topics in the CPLEX User's
    Manual.
    """
    continuous = CPX_CONTINUOUS
    binary = CPX_BINARY
    integer = CPX_INTEGER
    semi_integer = CPX_SEMIINT
    semi_continuous = CPX_SEMICONT

    def __getitem__(self, item: str) -> str:
        """Converts a constant to a string.

        Returns:
            Variable type name.

        Raises:
            QiskitOptimizationError: if the argument is not a valid type.

        Example usage:

        >>> from qiskit.optimization.problems import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> op.variables.type.binary
        'B'
        >>> op.variables.type['B']
        'binary'
        """
        if item == CPX_CONTINUOUS:
            return 'continuous'
        if item == CPX_BINARY:
            return 'binary'
        if item == CPX_INTEGER:
            return 'integer'
        if item == CPX_SEMIINT:
            return 'semi_integer'
        if item == CPX_SEMICONT:
            return 'semi_continuous'
        raise QiskitOptimizationError('Invalid variable type: {}'.format(item))


class VariablesInterface(BaseInterface):
    """Methods for adding, querying, and modifying variables.

    Example usage:

    >>> from qiskit.optimization import OptimizationProblem
    >>> op = OptimizationProblem()
    >>> indices = op.variables.add(names = ["x0", "x1", "x2"])
    >>> # default values for lower_bounds are 0.0
    >>> op.variables.get_lower_bounds()
    [0.0, 0.0, 0.0]
    >>> # values can be set either one at a time or many at a time
    >>> op.variables.set_lower_bounds(0, 1.0)
    >>> op.variables.set_lower_bounds([("x1", -1.0), (2, 3.0)])
    >>> # values can be queried as a range
    >>> op.variables.get_lower_bounds(0, "x1")
    [1.0, -1.0]
    >>> # values can be queried as a sequence in arbitrary order
    >>> op.variables.get_lower_bounds(["x1", "x2", 0])
    [-1.0, 3.0, 1.0]
    >>> # can query the number of variables
    >>> op.variables.get_num()
    3
    >>> op.variables.set_types(0, op.variables.type.binary)
    >>> op.variables.get_num_binary()
    1
    """

    type = VarTypes()

    def __init__(self):
        """Creates a new VariablesInterface.

        The variables interface is exposed by the top-level `OptimizationProblem` class
        as `OptimizationProblem.variables`.  This constructor is not meant to be used
        externally.
        """
        super(VariablesInterface, self).__init__()
        self._names = []
        self._lb = []
        self._ub = []
        self._types = []
        # self._obj = []
        # self._columns = []
        self._index = NameIndex()

    def get_num(self) -> int:
        """Returns the number of variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.continuous, t.binary, t.integer])
        >>> op.variables.get_num()
        3
        """
        return len(self._names)

    def get_num_continuous(self) -> int:
        """Returns the number of continuous variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.continuous, t.binary, t.integer])
        >>> op.variables.get_num_continuous()
        1
        """
        return self._types.count(VarTypes.continuous)

    def get_num_integer(self) -> int:
        """Returns the number of integer variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.continuous, t.binary, t.integer])
        >>> op.variables.get_num_integer()
        1
        """
        return self._types.count(VarTypes.integer)

    def get_num_binary(self) -> int:
        """Returns the number of binary variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.semi_continuous, t.binary, t.integer])
        >>> op.variables.get_num_binary()
        1
        """
        return self._types.count(VarTypes.binary)

    def get_num_semicontinuous(self) -> int:
        """Returns the number of semi-continuous variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.semi_continuous, t.semi_integer, t.semi_integer])
        >>> op.variables.get_num_semicontinuous()
        1
        """
        return self._types.count(VarTypes.semi_continuous)

    def get_num_semiinteger(self) -> int:
        """Returns the number of semi-integer variables in the problem.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.semi_continuous, t.semi_integer, t.semi_integer])
        >>> op.variables.get_num_semiinteger()
        2
        """
        return self._types.count(VarTypes.semi_integer)

    # pylint: disable=invalid-name
    def add(self, obj: None = None, lb: Optional[List[float]] = None,
            ub: Optional[List[float]] = None, types: str = "", names: Optional[List[str]] = None,
            columns: None = None) -> range:
        """Adds variables and related data to the problem.

        variables.add accepts the keyword arguments obj, lb, ub, types, names, and columns.
        If more than one argument is specified, all arguments must have the same length.

        Note
            `obj` and `columns` have not been supported yet.
            Use `objective` and `linear_constraint` instead.

        Args:
            lb: a list of floats specifying the lower bounds on the variables.

            ub: a list of floats specifying the upper bounds on the variables.

            types: must be either a list of single-character strings or a string containing
                the types of the variables.

                Note
                    If types is specified, the problem type will be a MIP, even if all variables are
                    specified to be continuous.

            names: a list of strings.

            obj: not supported by Qiskit Aqua. Use `objective` instead.

            columns: not supported by Qiskit Aqua. Use `linear_constraints` instead.

        Returns:
            an iterator containing the indices of the added variables.

        Raises:
            QiskitOptimizationError: if arguments are not valid.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> from cplex import SparsePair, infinity
        >>> op = OptimizationProblem()
        >>> indices = op.linear_constraints.add(names = ["c0", "c1", "c2"])
        >>> indices = op.variables.add(obj = [1.0, 2.0, 3.0],\
                                      types = [op.variables.type.integer] * 3)
        >>> indices = op.variables.add(obj = [1.0, 2.0, 3.0],\
                                      lb = [-1.0, 1.0, 0.0],\
                                      ub = [100.0, infinity, infinity],\
                                      types = [op.variables.type.integer] * 3,\
                                      names = ["0", "1", "2"],\
                                      columns = [SparsePair(ind = ['c0', 2], val = [1.0, -1.0]),\
                                      [['c2'],[2.0]],\
                                      SparsePair(ind = [0, 1], val = [3.0, 4.0])])

        >>> op.variables.get_lower_bounds()
        [0.0, 0.0, 0.0, -1.0, 1.0, 0.0]
        >>> op.variables.get_cols("1")
        SparsePair(ind = [2], val = [2.0])
        """
        if obj:
            raise QiskitOptimizationError("Please use ObjectiveInterface instead of obj.")
        if columns:
            raise QiskitOptimizationError(
                "Please use LinearConstraintInterface instead of columns.")

        start = self.get_num()
        arg_list = init_list_args(lb, ub, types, names)
        arg_lengths = [len(x) for x in arg_list]
        if len(arg_lengths) == 0:
            return range(start, start)
        max_length = max(arg_lengths)
        if max_length == 0:
            return range(start, start)
        for arg_length in arg_lengths:
            if arg_length > 0 and arg_length != max_length:
                raise QiskitOptimizationError("inconsistent arguments")

        lb = lb or [0] * max_length
        ub = ub or [infinity] * max_length
        types = types or [VarTypes.continuous] * max_length
        for i, t in enumerate(types):
            if t == VarTypes.binary and ub[i] == infinity:
                ub[i] = 1
        self._lb.extend(lb)
        self._ub.extend(ub)
        self._types.extend(types)

        names = names or [''] * max_length
        for i, name in enumerate(names):
            if name == '':
                names[i] = 'x' + str(start + i + 1)
        self._names.extend(names)
        self._index.build(self._names)

        return range(start, start + max_length)

    def delete(self, *args):
        """Deletes variables from the problem.

        There are four forms by which variables.delete may be called.

        variables.delete()
          deletes all variables from the problem.

        variables.delete(i)
          i must be a variable name or index. Deletes the variable
          whose index or name is i.

        variables.delete(s)
          s must be a sequence of variable names or indices. Deletes
          the variables with names or indices contained within s.
          Equivalent to [variables.delete(i) for i in s].

        variables.delete(begin, end)
          begin and end must be variable indices or variable names.
          Deletes the variables with indices between begin and end,
          inclusive of end. Equivalent to
          variables.delete(range(begin, end + 1)). This will give the
          best performance when deleting batches of variables.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names=[str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.variables.delete(8)
        >>> op.variables.get_names()
        ['0', '1', '2', '3', '4', '5', '6', '7', '9']
        >>> op.variables.delete("1", 3)
        >>> op.variables.get_names()
        ['0', '4', '5', '6', '7', '9']
        >>> op.variables.delete([2, "0", 5])
        >>> op.variables.get_names()
        ['4', '6', '7']
        >>> op.variables.delete()
        >>> op.variables.get_names()
        []
        """

        def _delete(i):
            del self._names[i]
            del self._ub[i]
            del self._lb[i]
            del self._types[i]

        if len(args) == 0:
            # Delete all
            self._names = []
            self._ub = []
            self._lb = []
            self._types = []
            # self._columns = []
            self._index = NameIndex()

        keys = self._index.convert(*args)
        if isinstance(keys, int):
            keys = [keys]
        for i in sorted(keys, reverse=True):
            _delete(i)
        self._index.build(self._names)

    def set_lower_bounds(self, *args):
        """Sets the lower bound for a variable or set of variables.

        There are two forms by which variables.set_lower_bounds may be
        called.

        variables.set_lower_bounds(i, lb)
          i must be a variable name or index and lb must be a real
          number.  Sets the lower bound of the variable whose index
          or name is i to lb.

        variables.set_lower_bounds(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, lb) pairs, each
          of which consists of a variable name or index and a real
          number.  Sets the lower bound of the specified variables to
          the corresponding values.  Equivalent to
          [variables.set_lower_bounds(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names = ["x0", "x1", "x2"])
        >>> op.variables.set_lower_bounds(0, 1.0)
        >>> op.variables.get_lower_bounds()
        [1.0, 0.0, 0.0]
        >>> op.variables.set_lower_bounds([(2, 3.0), ("x1", -1.0)])
        >>> op.variables.get_lower_bounds()
        [1.0, -1.0, 3.0]
        """

        def _set(i, v):
            self._lb[self._index.convert(i)] = v

        self._setter(_set, *args)

    def set_upper_bounds(self, *args):
        """Sets the upper bound for a variable or set of variables.

        There are two forms by which variables.set_upper_bounds may be
        called.

        variables.set_upper_bounds(i, ub)
          i must be a variable name or index and ub must be a real
          number.  Sets the upper bound of the variable whose index
          or name is i to ub.

        variables.set_upper_bounds(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, ub) pairs, each
          of which consists of a variable name or index and a real
          number.  Sets the upper bound of the specified variables to
          the corresponding values.  Equivalent to
          [variables.set_upper_bounds(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names = ["x0", "x1", "x2"])
        >>> op.variables.set_upper_bounds(0, 1.0)
        >>> op.variables.set_upper_bounds([("x1", 10.0), (2, 3.0)])
        >>> op.variables.get_upper_bounds()
        [1.0, 10.0, 3.0]
        """

        def _set(i, v):
            self._ub[self._index.convert(i)] = v

        self._setter(_set, *args)

    def set_names(self, *args):
        """Sets the name of a variable or set of variables.

        There are two forms by which variables.set_names may be
        called.

        variables.set_names(i, name)
          i must be a variable name or index and name must be a
          string.

        variables.set_names(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, name) pairs,
          each of which consists of a variable name or index and a
          string.  Sets the name of the specified variables to the
          corresponding strings.  Equivalent to
          [variables.set_names(pair[0], pair[1]) for pair in seq_of_pairs].

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(types = [t.continuous, t.binary, t.integer])
        >>> op.variables.set_names(0, "first")
        >>> op.variables.set_names([(2, "third"), (1, "second")])
        >>> op.variables.get_names()
        ['first', 'second', 'third']
        """

        def _set(i, v):
            self._names[self._index.convert(i)] = v

        self._setter(_set, *args)
        self._index.build(self._names)

    def set_types(self, *args):
        """Sets the type of a variable or set of variables.

        There are two forms by which variables.set_types may be
        called.

        variables.set_types(i, type)
          i must be a variable name or index and name must be a
          single-character string.

        variables.set_types(seq_of_pairs)
          seq_of_pairs must be a list or tuple of (i, type) pairs,
          each of which consists of a variable name or index and a
          single-character string.  Sets the type of the specified
          variables to the corresponding strings.  Equivalent to
          [variables.set_types(pair[0], pair[1]) for pair in seq_of_pairs].

        Note
          If the types are set, the problem will be treated as a MIP,
          even if all variable types are continuous.

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(5)])
        >>> op.variables.set_types(0, op.variables.type.continuous)
        >>> op.variables.set_types([("1", op.variables.type.integer),\
                                   ("2", op.variables.type.binary),\
                                   ("3", op.variables.type.semi_continuous),\
                                   ("4", op.variables.type.semi_integer)])
        >>> op.variables.get_types()
        ['C', 'I', 'B', 'S', 'N']
        >>> op.variables.type[op.variables.get_types(0)]
        'continuous'
        """

        def _set(i, v):
            if v not in [CPX_CONTINUOUS, CPX_BINARY, CPX_INTEGER, CPX_SEMICONT, CPX_SEMIINT]:
                raise QiskitOptimizationError(
                    "Second argument must be a string, as per VarTypes constants.")
            self._types[self._index.convert(i)] = v

        self._setter(_set, *args)

    def get_lower_bounds(self, *args) -> Union[float, List[float]]:
        """Returns the lower bounds on variables from the problem.

        There are four forms by which variables.get_lower_bounds may be called.

        variables.get_lower_bounds()
          return the lower bounds on all variables from the problem.

        variables.get_lower_bounds(i)
          i must be a variable name or index.  Returns the lower
          bound on the variable whose index or name is i.

        variables.get_lower_bounds(s)
          s must be a sequence of variable names or indices.  Returns
          the lower bounds on the variables with indices the members
          of s.  Equivalent to
          [variables.get_lower_bounds(i) for i in s]

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(lb = [1.5 * i for i in range(10)],\
                                      names = [str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.variables.get_lower_bounds(8)
        12.0
        >>> op.variables.get_lower_bounds("1",3)
        [1.5, 3.0, 4.5]
        >>> op.variables.get_lower_bounds([2,"0",5])
        [3.0, 0.0, 7.5]
        >>> op.variables.get_lower_bounds()
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """

        def _get(i):
            return self._lb[i]

        if len(args) == 0:
            return copy.deepcopy(self._lb)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_upper_bounds(self, *args) -> Union[float, List[float]]:
        """Returns the upper bounds on variables from the problem.

        There are four forms by which variables.get_upper_bounds may be called.

        variables.get_upper_bounds()
          return the upper bounds on all variables from the problem.

        variables.get_upper_bounds(i)
          i must be a variable name or index.  Returns the upper
          bound on the variable whose index or name is i.

        variables.get_upper_bounds(s)
          s must be a sequence of variable names or indices.  Returns
          the upper bounds on the variables with indices the members
          of s.  Equivalent to
          [variables.get_upper_bounds(i) for i in s]

        variables.get_upper_bounds(begin, end)
          begin and end must be variable indices or variable names.
          Returns the upper bounds on the variables with indices between
          begin and end, inclusive of end. Equivalent to
          variables.get_upper_bounds(range(begin, end + 1)).

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(ub = [(1.5 * i) + 1.0 for i in range(10)],\
                                      names = [str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.variables.get_upper_bounds(8)
        13.0
        >>> op.variables.get_upper_bounds("1",3)
        [2.5, 4.0, 5.5]
        >>> op.variables.get_upper_bounds([2,"0",5])
        [4.0, 1.0, 8.5]
        >>> op.variables.get_upper_bounds()
        [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5]
        """

        def _get(i):
            return self._ub[i]

        if len(args) == 0:
            return copy.deepcopy(self._ub)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_names(self, *args) -> Union[str, List[str]]:
        """Returns the names of variables from the problem.

        There are four forms by which variables.get_names may be called.

        variables.get_names()
          return the names of all variables from the problem.

        variables.get_names(i)
          i must be a variable index.  Returns the name of variable i.

        variables.get_names(s)
          s must be a sequence of variable indices.  Returns the
          names of the variables with indices the members of s.
          Equivalent to [variables.get_names(i) for i in s]

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names = ['x' + str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.variables.get_names(8)
        'x8'
        >>> op.variables.get_names(1,3)
        ['x1', 'x2', 'x3']
        >>> op.variables.get_names([2,0,5])
        ['x2', 'x0', 'x5']
        >>> op.variables.get_names()
        ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
        """

        def _get(i):
            return self._names[i]

        if len(args) == 0:
            return copy.deepcopy(self._names)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_types(self, *args) -> Union[str, List[str]]:
        """Returns the types of variables from the problem.

        There are four forms by which variables.types may be called.

        variables.types()
          return the types of all variables from the problem.

        variables.types(i)
          i must be a variable name or index.  Returns the type of
          the variable whose index or name is i.

        variables.types(s)
          s must be a sequence of variable names or indices.  Returns
          the types of the variables with indices the members of s.
          Equivalent to [variables.get_types(i) for i in s]

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> t = op.variables.type
        >>> indices = op.variables.add(names = [str(i) for i in range(5)],\
                                      types = [t.continuous, t.integer,\
                                      t.binary, t.semi_continuous, t.semi_integer])
        >>> op.variables.get_num()
        5
        >>> op.variables.get_types(3)
        'S'
        >>> op.variables.get_types(1,3)
        ['I', 'B', 'S']
        >>> op.variables.get_types([2,0,4])
        ['B', 'C', 'N']
        >>> op.variables.get_types()
        ['C', 'I', 'B', 'S', 'N']
        """

        def _get(i):
            return self._types[i]

        if len(args) == 0:
            return copy.deepcopy(self._types)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_cols(self, *args):
        """get_cols is not supported"""
        raise NotImplementedError("Please use LinearConstraintInterface instead.")

    def get_obj(self, *args):
        """get_obj is not supported"""
        raise NotImplementedError("Please use ObjectiveInterface instead.")

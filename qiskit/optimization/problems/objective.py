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
import numbers

from qiskit.optimization.utils import BaseInterface, QiskitOptimizationError
from qiskit.optimization.utils.helpers import listify, convert
from cplex import SparsePair
from cplex.exceptions import CplexSolverError

CPX_MAX = -1
CPX_MIN = 1


class ObjSense(object):
    """Constants defining the sense of the objective function."""
    maximize = CPX_MAX
    minimize = CPX_MIN

    def __getitem__(self, item):
        """Converts a constant to a string.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.sense.minimize
        1
        >>> op.objective.sense[1]
        'minimize'
        """
        if item == CPX_MAX:
            return 'maximize'
        if item == CPX_MIN:
            return 'minimize'


class ObjectiveInterface(BaseInterface):
    """Contains methods for querying and modifying the objective function."""

    sense = ObjSense()  # See `ObjSense()`

    def __init__(self, varsgetindexfunc=None):
        super(ObjectiveInterface, self).__init__()
        self._linear = {}
        self._quadratic = {}
        self._name = None
        self._sense = ObjSense.minimize
        self._offset = 0.0

        def defaultvarsgetindexfunc(i):
            if isinstance(i, int):
                return i
            else:
                raise ("Cannot convert variable names without appropriate vargetindexfunc!")

        if varsgetindexfunc:
            self._varsgetindexfunc = varsgetindexfunc
        else:
            self._varsgetindexfunc = defaultvarsgetindexfunc

    def set_linear(self, *args):
        """Changes the linear part of the objective function.

        Can be called by two forms:

        objective.set_linear(var, value)
          var must be a variable index or name and value must be a
          float.  Changes the coefficient of the variable identified
          by var to value.

        objective.set_linear(sequence)
          sequence is a sequence of pairs (var, value) as described
          above.  Changes the coefficients for the specified
          variables to the given values.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(4)])
        >>> op.objective.get_linear()
        [0.0, 0.0, 0.0, 0.0]
        >>> op.objective.set_linear(0, 1.0)
        >>> op.objective.get_linear()
        [1.0, 0.0, 0.0, 0.0]
        >>> op.objective.set_linear("3", -1.0)
        >>> op.objective.get_linear()
        [1.0, 0.0, 0.0, -1.0]
        >>> op.objective.set_linear([("2", 2.0), (1, 0.5)])
        >>> op.objective.get_linear()
        [1.0, 0.5, 2.0, -1.0]
        """

        def _set(i, v):
            try:
                v = v + 0.0
                if v == 0.0 and i in self._linear:
                    self._linear.pop(i)
                else:
                    self._linear[convert(i, self._varsgetindexfunc)] = v
            except CplexSolverError:
                raise QiskitOptimizationError(
                    "Value of a coefficient needs to allow for addition of a float!")

        # check for all elements in args whether they are types
        if len(args) == 1 and all(hasattr(el, '__len__') and len(el) == 2 for el in args[0]):
            for el in args[0]:
                _set(*el)
        elif len(args) == 2:
            _set(*args)
        else:
            raise QiskitOptimizationError("Invalid arguments to set set_linear!")

    def set_quadratic(self, *args):
        """Sets the quadratic part of the objective function.

        Call this method with a list with length equal to the number
        of variables in the problem.

        If the quadratic objective function is separable, the entries
        of the list must all be of type float, int, or long.

        If the quadratic objective function is not separable, the
        entries of the list must be either SparsePair instances or
        lists of two lists, the first of which contains variable
        indices or names, the second of which contains the values that
        those variables take.

        Note
          Successive calls to set_quadratic will overwrite any previous
          quadratic objective function.  To modify only part of the
          quadratic objective function, use the method
          set_quadratic_coefficients.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(3)])
        >>> op.objective.set_quadratic([SparsePair(ind = [0, 1, 2], val = [1.0, -2.0, 0.5]),\
                                       SparsePair(ind = [0, 1], val = [-2.0, -1.0]),\
                                       SparsePair(ind = [0, 2], val = [0.5, -3.0])])
        >>> op.objective.get_quadratic()
        [SparsePair(ind = [0, 1, 2], val = [1.0, -2.0, 0.5]),
            SparsePair(ind = [0, 1], val = [-2.0, -1.0]),
                SparsePair(ind = [0, 2], val = [0.5, -3.0])]
        >>> op.objective.set_quadratic([1.0, 2.0, 3.0])
        >>> op.objective.get_quadratic()
            [SparsePair(ind = [0], val = [1.0]), SparsePair(ind = [1], val = [2.0]),
                SparsePair(ind = [2], val = [3.0])]
        """

        # clear data
        self._quadratic = {}

        def _set(i, j, val):
            i = convert(i, self._varsgetindexfunc)
            j = convert(j, self._varsgetindexfunc)
            i_vals = self._quadratic.setdefault(i, {})
            j_vals = self._quadratic.setdefault(j, {})
            # NOTE: The following check is not necessary, considering we clear the _quadratic first
            # if i in j_vals.keys() and j_vals[i] != val or j in i_vals.keys() and i_vals[j] != val:
            #        raise QiskitOptimizationError('Q is not symmetric in set_quadratic.')
            i_vals[j] = val
            j_vals[i] = val

        if len(args) != 1:
            raise QiskitOptimizationError("set_quadratic expects one argument, which is a list")
        if args[0] and isinstance(args[0][0], numbers.Number):
            for i, val in enumerate(args[0]):
                _set(i, i, val)
        else:
            for i, sp in enumerate(args[0]):
                if isinstance(sp, SparsePair):
                    for j, val in zip(sp.ind, sp.val):
                        _set(i, j, val)
                elif isinstance(sp, Sequence) and len(sp) == 2:
                    for i, (j, val) in enumerate(zip(sp[0], sp[1])):
                        _set(i, j, val)
                else:
                    QiskitOptimizationError(
                        "set_quadratic expects a list of the length equal to the number of " +
                        "variables, where each entry has a pair of the indices of the other " +
                        "variables and values, or the corresponding SparsePair")

    def set_quadratic_coefficients(self, *args):
        """Sets coefficients of the quadratic component of the objective function.

        To set a single coefficient, call this method as

        objective.set_quadratic_coefficients(v1, v2, val)

        where v1 and v2 are names or indices of variables and val is
        the value for the coefficient.

        To set multiple coefficients, call this method as

        objective.set_quadratic_coefficients(sequence)

        where sequence is a list or tuple of triples (v1, v2, val) as
        described above.

        Note
          Since the quadratic objective function must be symmetric, each
          triple in which v1 is different from v2 is used to set both
          the (v1, v2) coefficient and the (v2, v1) coefficient.  If
          (v1, v2) and (v2, v1) are set with a single call, the second
          value is stored.

        Note
          Attempting to set many coefficients with set_quadratic_coefficients
          can be time consuming. Instead, use the method set_quadratic to set
          the quadratic part of the objective efficiently.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(3)])
        >>> op.objective.set_quadratic_coefficients(0, 1, 1.0)
        >>> op.objective.get_quadratic()
        [SparsePair(ind = [1], val = [1.0]), SparsePair(ind = [0], val = [1.0]),
            SparsePair(ind = [], val = [])]
        >>> op.objective.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        >>> op.objective.get_quadratic()
        [SparsePair(ind = [1, 2], val = [1.0, 3.0]), SparsePair(ind = [0, 1], val = [1.0, 2.0]),
            SparsePair(ind = [0], val = [3.0])]
        >>> op.objective.set_quadratic_coefficients([(0, 1, 4.0), (1, 0, 5.0)])
        >>> op.objective.get_quadratic()
        [SparsePair(ind = [1, 2], val = [5.0, 3.0]), SparsePair(ind = [0, 1], val = [5.0, 2.0]),
            SparsePair(ind = [0], val = [3.0])]
        """

        def _set(i, j, val):
            i = convert(i, self._varsgetindexfunc)
            j = convert(j, self._varsgetindexfunc)
            i_vals = self._quadratic.setdefault(i, {})
            j_vals = self._quadratic.setdefault(j, {})
            if val == 0:
                if j in i_vals:
                    i_vals.pop(j)
                if i != j and i in j_vals:
                    j_vals.pop(i)
            else:
                i_vals[j] = val
                j_vals[i] = val

        if len(args) not in (3, 1):
            raise QiskitOptimizationError("Wrong number of arguments")
        if isinstance(args[0], (str, int)):
            arg_list = [args]
        else:
            arg_list = args[0]
        for i, j, val in arg_list:
            _set(i, j, val)

    def set_sense(self, sense):
        """Sets the sense of the objective function.

        The argument to this method must be either
        objective.sense.minimize or objective.sense.maximize.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.sense[op.objective.get_sense()]
        'minimize'
        >>> op.objective.set_sense(op.objective.sense.maximize)
        >>> op.objective.sense[op.objective.get_sense()]
        'maximize'
        >>> op.objective.set_sense(op.objective.sense.minimize)
        >>> op.objective.sense[op.objective.get_sense()]
        'minimize'
        """
        if sense in [CPX_MAX, CPX_MIN]:
            self._sense = sense
        else:
            raise QiskitOptimizationError(
                "sense should be one of [CPX_MAX, CPX_MIN], i.e., objective.sense.minimize or " +
                "objective.sense.maximize.")

    def set_name(self, name):
        """Sets the name of the objective function.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.set_name("cost")
        >>> op.objective.get_name()
        'cost'
        """
        self._name = name

    def get_linear(self, *args):
        """Returns the linear coefficients of a set of variables.

        Can be called by four forms.

        objective.get_linear()
          return the linear objective coefficients of all variables
          from the problem.

        objective.get_linear(i)
          i must be a variable name or index.  Returns the linear
          objective coefficient of the variable whose index or name
          is i.

        objective.get_linear(s)
          s must be a sequence of variable names or indices.  Returns
          the linear objective coefficient of the variables with
          indices the members of s.  Equivalent to
          [objective.get_linear(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(obj = [1.5 * i for i in range(10)],\
                            names = [str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.objective.get_linear(8)
        12.0
        >>> op.objective.get_linear([2,"0",5])
        [3.0, 0.0, 7.5]
        >>> op.objective.get_linear()
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """

        def _get(i):
            i = convert(i, self._varsgetindexfunc)
            return self._linear.get(i, 0.0)

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._linear)
        elif len(args) == 1:
            if isinstance(args[0], str):
                return _get(args[0])
            elif isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            for i in args:
                out.append(_get(i))
        return out

    def get_quadratic(self, *args):
        """Returns a set of columns of the quadratic component of the objective function.

        Returns a SparsePair instance or a list of SparsePair instances.

        Can be called by four forms.

        objective.get_quadratic()
          return the entire quadratic objective function.

        objective.get_quadratic(i)
          i must be a variable name or index.  Returns the column of
          the quadratic objective function associated with the
          variable whose index or name is i.

        objective.get_quadratic(s)
          s must be a sequence of variable names or indices.  Returns
          the columns of the quadratic objective function associated
          with the variables with indices the members of s.
          Equivalent to [objective.get_quadratic(i) for i in s]

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(10)])
        >>> op.variables.get_num()
        10
        >>> op.objective.set_quadratic([1.5 * i for i in range(10)])
        >>> op.objective.get_quadratic(8)
        SparsePair(ind = [8], val = [12.0])
        >>> op.objective.get_quadratic([3,"1",5])
        [SparsePair(ind = [3], val = [4.5]), SparsePair(ind = [1], val = [1.5]),
            SparsePair(ind = [5], val = [7.5])]
        >>> op.objective.get_quadratic()
        [SparsePair(ind = [], val = []), SparsePair(ind = [1], val = [1.5]),
            SparsePair(ind = [2], val = [3.0]), SparsePair(ind = [3], val = [4.5]),
            SparsePair(ind = [4], val = [6.0]), SparsePair(ind = [5], val = [7.5]),
            SparsePair(ind = [6], val = [9.0]), SparsePair(ind = [7], val = [10.5]),
            SparsePair(ind = [8], val = [12.0]), SparsePair(ind = [9], val = [13.5])]
        """

        def _get(i):
            i = convert(i, self._varsgetindexfunc)
            return SparsePair(list(self._quadratic[i].keys()), list(self._quadratic[i].values()))

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._quadratic)
        elif len(args) == 1:
            if isinstance(args[0], str):
                return _get(args[0])
            elif isinstance(args[0], Sequence):
                for i in args[0]:
                    out.append(_get(i))
            else:
                return _get(args[0])
        else:
            # NOTE: This is not documented, but perhaps useful.
            for i in args:
                out.append(_get(i))
        return out

    def get_quadratic_coefficients(self, *args):
        """Returns individual coefficients from the quadratic objective function.

        To query a single coefficient, call this as

        objective.get_quadratic_coefficients(v1, v2)

        where v1 and v2 are indices or names of variables.

        To query multiple coefficients, call this method as

        objective.get_quadratic_coefficients(sequence)

        where sequence is a list or tuple of pairs (v1, v2) as
        described above.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(3)])
        >>> op.objective.set_quadratic_coefficients(0, 1, 1.0)
        >>> op.objective.get_quadratic_coefficients("1", 0)
        1.0
        >>> op.objective.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0), (1, 0, 5.0)])
        >>> op.objective.get_quadratic_coefficients([(1, 0), (1, "1"), (2, "0")])
        [5.0, 2.0, 3.0]
        """

        def _get(i, j):
            i = convert(i, self._varsgetindexfunc)
            j = convert(j, self._varsgetindexfunc)
            return self._quadratic.get(i, {}).get(j, 0)

        out = []
        if len(args) == 0:
            return copy.deepcopy(self._quadratic)
        elif len(args) == 2:
            return _get(args[0], args[1])
        elif len(args) == 1:
            if isinstance(args[0], str):
                raise QiskitOptimizationError('Incompatible type: %s' % args[0])
            elif isinstance(args[0], Sequence):
                for (i, j) in args[0]:
                    out.append(_get(i, j))
                return out
            else:
                raise QiskitOptimizationError(
                    "get_quadratic_coefficients passed a single argument that is not iterable.")
        else:
            raise QiskitOptimizationError(
                "get_quadratic_coefficients expects either values two indices as two arguments, " +
                "or a sequence of tuples.")

    def get_sense(self):
        """Returns the sense of the objective function.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.sense[op.objective.get_sense()]
        1
        >>> op.objective.set_sense(op.objective.sense.maximize)
        >>> op.objective.sense[op.objective.get_sense()]
        -1
        >>> op.objective.set_sense(op.objective.sense.minimize)
        >>> op.objective.sense[op.objective.get_sense()]
        1
        """
        return self._sense

    def get_name(self):
        """Returns the name of the objective function.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.set_name("cost")
        >>> op.objective.get_name()
        'cost'
        """
        return self._name

    def get_num_quadratic_variables(self):
        """Returns the number of variables with quadratic coefficients.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(3)])
        >>> op.objective.set_quadratic_coefficients(0, 1, 1.0)
        >>> op.objective.get_num_quadratic_variables()
        2
        >>> op.objective.set_quadratic([1.0, 0.0, 0.0])
        >>> op.objective.get_num_quadratic_variables()
        1
        >>> op.objective.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        >>> op.objective.get_num_quadratic_variables()
        3
        """
        return len(self._quadratic.items())

    def get_num_quadratic_nonzeros(self):
        """Returns the number of nonzeros in the quadratic objective function.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(3)])
        >>> op.objective.set_quadratic_coefficients(0, 1, 1.0)
        >>> op.objective.get_num_quadratic_nonzeros()
        2
        >>> op.objective.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        >>> op.objective.get_num_quadratic_nonzeros()
        5
        >>> op.objective.set_quadratic_coefficients([(0, 1, 4.0), (1, 0, 0.0)])
        >>> op.objective.get_num_quadratic_nonzeros()
        3
        """
        nnz = 0
        for (i, v) in self._quadratic.items():
            nnz += len(v)
        return nnz

    def get_offset(self):
        """Returns the constant offset of the objective function for a problem.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> offset = op.objective.get_offset()
        >>> abs(offset - 0.0) < 1e-6
        True
        """
        return self._offset

    def set_offset(self, offset):
        """Sets the constant offset of the objective function for a problem.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> op.objective.set_offset(3.14)
        >>> offset = op.objective.get_offset()
        >>> abs(offset - 3.14) < 1e-6
        True
        """
        self._offset = offset

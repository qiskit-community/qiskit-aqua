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
from logging import getLogger
from typing import List, Dict, Tuple

from cplex import SparsePair, SparseTriple

from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.helpers import convert, NameIndex
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError

logger = getLogger(__name__)


class QuadraticConstraintInterface(BaseInterface):
    """Methods for adding, modifying, and querying quadratic constraints."""

    def __init__(self, varsgetindexfunc=None):
        """Creates a new QuadraticConstraintInterface.

        The quadratic constraints interface is exposed by the top-level
        `OptimizationProblem` class as `OptimizationProblem.quadratic_constraints`.  This constructor
        is not meant to be used externally.
        """
        super(QuadraticConstraintInterface, self).__init__()
        self._rhs = []
        self._senses = []
        self._names = []
        self._lin_expr: List[Dict[int, float]] = []
        self._quad_expr: List[Dict[Tuple[int, int], float]] = []
        self._name_index = NameIndex()
        self._varsgetindexfunc = varsgetindexfunc

    def get_num(self) -> int:
        """Returns the number of quadratic constraints.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ['x','y'])
        >>> l = SparsePair(ind = ['x'], val = [1.0])
        >>> q = SparseTriple(ind1 = ['x'], ind2 = ['y'], val = [1.0])
        >>> [op.quadratic_constraints.add(name=str(i), lin_expr=l, quad_expr=q)
        ...  for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        """
        return len(self._names)

    def add(self, lin_expr=None, quad_expr=None, sense="L", rhs=0.0, name=""):
        """Adds a quadratic constraint to the problem.

        Takes up to five keyword arguments:

        lin_expr : either a SparsePair or a list of two lists specifying
        the linear component of the constraint.

        Note
          lin_expr must not contain duplicate indices.  If lin_expr
          references a variable more than once, either by index, name,
          or a combination of index and name, an exception will be
          raised.

        quad_expr : either a SparseTriple or a list of three lists
        specifying the quadratic component of the constraint.

        Note
          quad_expr must not contain duplicate indices.  If quad_expr
          references a matrix entry more than once, either by indices,
          names, or a combination of indices and names, an exception
          will be raised.

        sense : either "L", "G", or "E"

        rhs : a float specifying the righthand side of the constraint.

        name : the name of the constraint.

        Returns the index of the added quadratic constraint.

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ['x','y'])
        >>> l = SparsePair(ind = ['x'], val = [1.0])
        >>> q = SparseTriple(ind1 = ['x'], ind2 = ['y'], val = [1.0])
        >>> op.quadratic_constraints.add(name = "my_quad",
        ...                             lin_expr = l,
        ...                             quad_expr = q,
        ...                             rhs = 1.0,
        ...                             sense = "G")
        0
        """
        # We only ever create one quadratic constraint at a time.

        # check constraint name
        if name == '':
            name = 'q{}'.format(len(self._names))
        if name in self._name_index:
            raise QiskitOptimizationError('Duplicate quadratic constraint name: {}'.format(name))
        self._names.append(name)

        # linear terms
        lin_expr_dict = {}
        if lin_expr is None:
            ind, val = [], []
        elif isinstance(lin_expr, SparsePair):
            ind, val = lin_expr.ind, lin_expr.val
        elif isinstance(lin_expr, Sequence):
            if len(lin_expr) != 2 or len(lin_expr[0]) != len(lin_expr[1]):
                raise QiskitOptimizationError('Invalid lin_expr: {}'.format(lin_expr))
            ind, val = lin_expr
        else:
            raise QiskitOptimizationError('Invalid lin_expr: {}'.format(lin_expr))
        for i, val in zip(ind, val):
            i2 = convert(i, self._varsgetindexfunc)
            if i2 in lin_expr_dict:
                logger.warning('lin_expr contains duplicate index: {}'.format(i))
            lin_expr_dict[i2] = val
        self._lin_expr.append(lin_expr_dict)

        # quadratic terms
        quad_expr_dict = {}
        if quad_expr is None:
            ind1, ind2, val = [], [], []
        elif isinstance(quad_expr, SparseTriple):
            ind1, ind2, val = quad_expr.ind1, quad_expr.ind2, quad_expr.val
        elif isinstance(quad_expr, Sequence):
            if len(quad_expr) != 3 or len(quad_expr[0]) != len(quad_expr[1]) or \
                    len(quad_expr[1]) != len(quad_expr[2]):
                raise QiskitOptimizationError('Invalid quad_expr: {}'.format(quad_expr))
            ind1, ind2, val = quad_expr
        else:
            raise QiskitOptimizationError('Invalid quad_expr: {}'.format(quad_expr))
        for i, j, val in zip(ind1, ind2, val):
            i2 = convert(i, self._varsgetindexfunc)
            j2 = convert(j, self._varsgetindexfunc)
            if i2 < j2:
                i2, j2 = j2, i2
            if (i2, j2) in quad_expr_dict:
                logger.warning('quad_expr contains duplicate index: {} {}'.format(i, j))
            quad_expr_dict[i2, j2] = val
        self._quad_expr.append(quad_expr_dict)

        if sense not in ['L', 'G', 'E']:
            raise QiskitOptimizationError('Invalid sense: {}'.format(sense))
        else:
            self._senses.append(sense)
        self._rhs.append(rhs)

        return self._name_index.convert(name)

    def delete(self, *args):
        """Deletes quadratic constraints from the problem.

        There are four forms by which quadratic_constraints.delete may be
        called.

        quadratic_constraints.delete()
          deletes all quadratic constraints from the problem.

        quadratic_constraints.delete(i)
          i must be a quadratic constraint name or index. Deletes
          the quadratic constraint whose index or name is i.

        quadratic_constraints.delete(s)
          s must be a sequence of quadratic constraint names or
          indices. Deletes the quadratic constraints with names or
          indices contained within s. Equivalent to
          [quadratic_constraints.delete(i) for i in s].

        quadratic_constraints.delete(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Deletes the quadratic constraints with
          indices between begin and end, inclusive of end. Equivalent to
          quadratic_constraints.delete(range(begin, end + 1)). This will
          give the best performance when deleting batches of quadratic
          constraints.

        See CPXdelqconstrs in the Callable Library Reference Manual for
        more detail.

        Example usage:

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names=['x', 'y'])
        >>> l = SparsePair(ind=['x'], val=[1.0])
        >>> q = SparseTriple(ind1=['x'], ind2=['y'], val=[1.0])
        >>> [op.quadratic_constraints.add(
        ...      name=str(i), lin_expr=l, quad_expr=q)
        ...  for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.delete(8)
        >>> op.quadratic_constraints.get_names()
        ['0', '1', '2', '3', '4', '5', '6', '7', '9']
        >>> op.quadratic_constraints.delete("1", 3)
        >>> op.quadratic_constraints.get_names()
        ['0', '4', '5', '6', '7', '9']
        >>> op.quadratic_constraints.delete([2, "0", 5])
        >>> op.quadratic_constraints.get_names()
        ['4', '6', '7']
        >>> op.quadratic_constraints.delete()
        >>> op.quadratic_constraints.get_names()
        []
        """
        if len(args) == 0:
            # delete all
            self._rhs = []
            self._senses = []
            self._names = []
            self._lin_expr = []
            self._quad_expr = []
            self._name_index = NameIndex()
            return
        elif len(args) == 1:
            # one item or sequence
            keys = self._name_index.convert(args[0])
            if isinstance(keys, int):
                keys = [keys]
        elif len(args) == 2:
            # begin and end of a range
            begin = self._name_index.convert(args[0])
            end = self._name_index.convert(args[1]) + 1
            keys = range(begin, end)
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))

        for i in sorted(keys, reverse=True):
            del self._rhs[i]
            del self._senses[i]
            del self._names[i]
            del self._lin_expr[i]
            del self._quad_expr[i]
        self._name_index.build(self._names)

    def get_rhs(self, *args):
        """Returns the righthand side of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_rhs()
          return the righthand side of all quadratic constraints
          from the problem.

        quadratic_constraints.get_rhs(i)
          i must be a quadratic constraint name or index.  Returns the
          righthand side of the quadratic constraint whose index or
          name is i.

        quadratic_constraints.get_rhs(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the righthand side of the quadratic
          constraints with indices the members of s.  Equivalent to
          [quadratic_constraints.get_rhs(i) for i in s]

        quadratic_constraints.get_rhs(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the righthand side of the quadratic
          constraints with indices between begin and end, inclusive of
          end. Equivalent to
          quadratic_constraints.get_rhs(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(10)])
        >>> [op.quadratic_constraints.add(rhs=1.5 * i, name=str(i))
        ...  for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_rhs(8)
        12.0
        >>> op.quadratic_constraints.get_rhs("1",3)
        [1.5, 3.0, 4.5]
        >>> op.quadratic_constraints.get_rhs([2,"0",5])
        [3.0, 0.0, 7.5]
        >>> op.quadratic_constraints.get_rhs()
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """
        if len(args) == 0:
            return self._rhs
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return self._rhs[keys]
        return [self._rhs[k] for k in keys]

    def get_senses(self, *args):
        """Returns the senses of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_senses()
          return the senses of all quadratic constraints from the
          problem.

        quadratic_constraints.get_senses(i)
          i must be a quadratic constraint name or index.  Returns the
          sense of the quadratic constraint whose index or name is i.

        quadratic_constraints.get_senses(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the senses of the quadratic constraints
          with indices the members of s.  Equivalent to
          [quadratic_constraints.get_senses(i) for i in s]

        quadratic_constraints.get_senses(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the senses of the quadratic
          constraints with indices between begin and end, inclusive of
          end. Equivalent to
          quadratic_constraints.get_senses(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = ["x0"])
        >>> [op.quadratic_constraints.add(name=str(i), sense=j)
        ...  for i, j in enumerate("GGLL")]
        [0, 1, 2, 3]
        >>> op.quadratic_constraints.get_num()
        4
        >>> op.quadratic_constraints.get_senses(1)
        'G'
        >>> op.quadratic_constraints.get_senses("1",3)
        ['G', 'L', 'L']
        >>> op.quadratic_constraints.get_senses([2,"0",1])
        ['L', 'G', 'G']
        >>> op.quadratic_constraints.get_senses()
        ['G', 'G', 'L', 'L']
        """
        if len(args) == 0:
            return self._senses
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return self._senses[keys]
        return [self._senses[k] for k in keys]

    def get_linear_num_nonzeros(self, *args):
        """Returns the number of nonzeros in the linear part of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_linear_num_nonzeros()
          return the number of nonzeros in all quadratic constraints
          from the problem.

        quadratic_constraints.get_linear_num_nonzeros(i)
          i must be a quadratic constraint name or index.  Returns the
          number of nonzeros in the quadratic constraint whose index
          or name is i.

        quadratic_constraints.get_linear_num_nonzeros(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the number of nonzeros in the quadratic
          constraints with indices the members of s.  Equivalent to
          [quadratic_constraints.get_linear_num_nonzeros(i) for i in s]

        quadratic_constraints.get_linear_num_nonzeros(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the number of nonzeros in the
          quadratic constraints with indices between begin and end,
          inclusive of end. Equivalent to
          quadratic_constraints.get_linear_num_nonzeros(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(11)], types = "B" * 11)
        >>> [op.quadratic_constraints.add(
        ...      name = str(i),
        ...      lin_expr = [range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_linear_num_nonzeros(8)
        8
        >>> op.quadratic_constraints.get_linear_num_nonzeros("1",3)
        [1, 2, 3]
        >>> op.quadratic_constraints.get_linear_num_nonzeros([2,"0",5])
        [2, 0, 5]
        >>> op.quadratic_constraints.get_linear_num_nonzeros()
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """

        def _nonzero(tab: Dict[int, float]) -> int:
            return len([0 for v in tab.values() if v != 0.0])

        if len(args) == 0:
            keys = range(self.get_num())
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return _nonzero(self._lin_expr[keys])
        return [_nonzero(self._lin_expr[k]) for k in keys]

    def get_linear_components(self, *args):
        """Returns the linear part of a set of quadratic constraints.

        Returns a list of SparsePair instances or one SparsePair instance.

        Can be called by four forms.

        quadratic_constraints.get_linear_components()
          return the linear components of all quadratic constraints
          from the problem.

        quadratic_constraints.get_linear_components(i)
          i must be a quadratic constraint name or index.  Returns the
          linear component of the quadratic constraint whose index or
          name is i.

        quadratic_constraints.get_linear_components(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the linear components of the quadratic
          constraints with indices the members of s.  Equivalent to
          [quadratic_constraints.get_linear_components(i) for i in s]

        quadratic_constraints.get_linear_components(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the linear components of the
          quadratic constraints with indices between begin and end,
          inclusive of end. Equivalent to
          quadratic_constraints.get_linear_components(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(11)], types = "B" * 11)
        >>> [op.quadratic_constraints.add(
        ...      name = str(i),
        ...      lin_expr = [range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(10)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_linear_components(8)
        SparsePair(ind = [0, 1, 2, 3, 4, 5, 6, 7], val = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        >>> op.quadratic_constraints.get_linear_components("1",3)
        [SparsePair(ind = [0], val = [1.0]), SparsePair(ind = [0, 1], val = [1.0, 2.0]), SparsePair(ind = [0, 1, 2], val = [1.0, 2.0, 3.0])]
        >>> op.quadratic_constraints.get_linear_components([2,"0",5])
        [SparsePair(ind = [0, 1], val = [1.0, 2.0]), SparsePair(ind = [], val = []), SparsePair(ind = [0, 1, 2, 3, 4], val = [1.0, 2.0, 3.0, 4.0, 5.0])]
        >>> op.quadratic_constraints.delete(4,9)
        >>> op.quadratic_constraints.get_linear_components()
        [SparsePair(ind = [], val = []), SparsePair(ind = [0], val = [1.0]), SparsePair(ind = [0, 1], val = [1.0, 2.0]), SparsePair(ind = [0, 1, 2], val = [1.0, 2.0, 3.0])]
        """

        def _linear_component(tab: Dict[int, float]) -> SparsePair:
            return SparsePair(ind=tuple(tab.keys()), val=tuple(tab.values()))

        if len(args) == 0:
            keys = range(self.get_num())
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return _linear_component(self._lin_expr[keys])
        return [_linear_component(self._lin_expr[k]) for k in keys]

    def get_quad_num_nonzeros(self, *args):
        """Returns the number of nonzeros in the quadratic part of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_quad_num_nonzeros()
          Returns the number of nonzeros in all quadratic constraints
          from the problem.

        quadratic_constraints.get_quad_num_nonzeros(i)
          i must be a quadratic constraint name or index.  Returns the
          number of nonzeros in the quadratic constraint whose index
          or name is i.

        quadratic_constraints.get_quad_num_nonzeros(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the number of nonzeros in the quadratic
          constraints with indices the members of s.  Equivalent to
          [quadratic_constraints.get_quad_num_nonzeros(i) for i in s]

        quadratic_constraints.get_quad_num_nonzeros(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the number of nonzeros in the
          quadratic constraints with indices between begin and end,
          inclusive of end. Equivalent to
          quadratic_constraints.get_quad_num_nonzeros(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(11)])
        >>> [op.quadratic_constraints.add(
        ...      name = str(i),
        ...      quad_expr = [range(i), range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(1, 11)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_quad_num_nonzeros(8)
        9
        >>> op.quadratic_constraints.get_quad_num_nonzeros("1",2)
        [1, 2, 3]
        >>> op.quadratic_constraints.get_quad_num_nonzeros([2,"1",5])
        [3, 1, 6]
        >>> op.quadratic_constraints.get_quad_num_nonzeros()
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """

        def _nonzero(tab: Dict[Tuple[int, int], float]) -> int:
            return len([0 for v in tab.values() if v != 0.0])

        if len(args) == 0:
            keys = range(self.get_num())
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return _nonzero(self._quad_expr[keys])
        return [_nonzero(self._quad_expr[k]) for k in keys]

    def get_quadratic_components(self, *args):
        """Returns the quadratic part of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_quadratic_components()
          return the quadratic components of all quadratic constraints
          from the problem.

        quadratic_constraints.get_quadratic_components(i)
          i must be a quadratic constraint name or index.  Returns the
          quadratic component of the quadratic constraint whose index or
          name is i.

        quadratic_constraints.get_quadratic_components(s)
          s must be a sequence of quadratic constraint names or
          indices.  Returns the quadratic components of the quadratic
          constraints with indices the members of s.  Equivalent to
          [quadratic_constraints.get_quadratic_components(i) for i in s]

        quadratic_constraints.get_quadratic_components(begin, end)
          begin and end must be quadratic constraint indices or quadratic
          constraint names. Returns the quadratic components of the
          quadratic constraints with indices between begin and end,
          inclusive of end. Equivalent to
          quadratic_constraints.get_quadratic_components(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(11)], types = "B" * 11)
        >>> [op.quadratic_constraints.add(
        ...      name = str(i),
        ...      quad_expr = [range(i), range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(1, 11)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_quadratic_components(8)
        SparseTriple(ind1 = [0, 1, 2, 3, 4, 5, 6, 7, 8], ind2 = [0, 1, 2, 3, 4, 5, 6, 7, 8], val = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        >>> op.quadratic_constraints.get_quadratic_components("1",3)
        [SparseTriple(ind1 = [0], ind2 = [0], val = [1.0]), SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0]), SparseTriple(ind1 = [0, 1, 2], ind2 = [0, 1, 2], val = [1.0, 2.0, 3.0]), SparseTriple(ind1 = [0, 1, 2, 3], ind2 = [0, 1, 2, 3], val = [1.0, 2.0, 3.0, 4.0])]
        >>> op.quadratic_constraints.get_quadratic_components([2,"1",5])
        [SparseTriple(ind1 = [0, 1, 2], ind2 = [0, 1, 2], val = [1.0, 2.0, 3.0]), SparseTriple(ind1 = [0], ind2 = [0], val = [1.0]), SparseTriple(ind1 = [0, 1, 2, 3, 4, 5], ind2 = [0, 1, 2, 3, 4, 5], val = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        >>> op.quadratic_constraints.delete(4,9)
        >>> op.quadratic_constraints.get_quadratic_components()
        [SparseTriple(ind1 = [0], ind2 = [0], val = [1.0]), SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0]), SparseTriple(ind1 = [0, 1, 2], ind2 = [0, 1, 2], val = [1.0, 2.0, 3.0]), SparseTriple(ind1 = [0, 1, 2, 3], ind2 = [0, 1, 2, 3], val = [1.0, 2.0, 3.0, 4.0])]
        """

        def _quadratic_component(tab: Dict[Tuple[int, int], float]) -> SparseTriple:
            ind1, ind2 = zip(*tab.keys())
            return SparseTriple(ind1=ind1, ind2=ind2, val=tuple(tab.values()))

        if len(args) == 0:
            keys = range(self.get_num())
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return _quadratic_component(self._quad_expr[keys])
        return [_quadratic_component(self._quad_expr[k]) for k in keys]

    def get_names(self, *args):
        """Returns the names of a set of quadratic constraints.

        Can be called by four forms.

        quadratic_constraints.get_names()
          return the names of all quadratic constraints from the
          problem.

        quadratic_constraints.get_names(i)
          i must be a quadratic constraint index.  Returns the name
          of constraint i.

        quadratic_constraints.get_names(s)
          s must be a sequence of quadratic constraint indices.
          Returns the names of the quadratic constraints with indices
          the members of s.  Equivalent to
          [quadratic_constraints.get_names(i) for i in s]

        quadratic_constraints.get_names(begin, end)
          begin and end must be quadratic constraint indices. Returns
          the names of the quadratic constraints with indices between
          begin and end, inclusive of end. Equivalent to
          quadratic_constraints.get_names(range(begin, end + 1)).

        >>> op = qiskit.optimization.OptimizationProblem()
        >>> indices = op.variables.add(names = [str(i) for i in range(11)])
        >>> [op.quadratic_constraints.add(
        ...      name = "q" + str(i),
        ...      quad_expr = [range(i), range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(1, 11)]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> op.quadratic_constraints.get_num()
        10
        >>> op.quadratic_constraints.get_names(8)
        'q9'
        >>> op.quadratic_constraints.get_names(1, 3)
        ['q2', 'q3', 'q4']
        >>> op.quadratic_constraints.get_names([2, 0, 5])
        ['q3', 'q1', 'q6']
        >>> op.quadratic_constraints.get_names()
        ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
        """
        if len(args) == 0:
            return self._names
        elif len(args) == 1:
            keys = self._name_index.convert(args[0])
        elif len(args) == 2:
            keys = self._name_index.convert(range(*args))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))
        if isinstance(keys, int):
            return self._names[keys]
        return [self._names[k] for k in keys]

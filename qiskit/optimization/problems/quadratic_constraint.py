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

"""Quadratic constraints interface"""

import copy
from collections.abc import Sequence
from logging import getLogger
from typing import List, Dict, Callable, Union, Optional

from cplex import SparsePair, SparseTriple
from scipy.sparse import dok_matrix

from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.helpers import NameIndex
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError

logger = getLogger(__name__)


class QuadraticConstraintInterface(BaseInterface):
    """Methods for adding, modifying, and querying quadratic constraints."""

    def __init__(self, varindex: Callable):
        """Creates a new QuadraticConstraintInterface.

        The quadratic constraints interface is exposed by the top-level
        `QuadraticProgram` class as `QuadraticProgram.quadratic_constraints`.
        This constructor is not meant to be used externally.
        """
        super(QuadraticConstraintInterface, self).__init__()
        self._rhs = []
        self._senses = []
        self._names = []
        self._lin_expr: List[Dict[int, float]] = []
        self._quad_expr: List[dok_matrix] = []
        self._index = NameIndex()
        self._varindex = varindex

    def get_num(self) -> int:
        """Returns the number of quadratic constraints.

        Example usage:

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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

    def add(self, lin_expr: Optional[SparsePair] = None, quad_expr: Optional[SparseTriple] = None,
            sense: str = "L", rhs: float = 0.0, name: str = "") -> int:
        """Adds a quadratic constraint to the problem.

        Takes up to following five keyword arguments.

        Args:
            lin_expr: either a SparsePair or a list of two lists specifying
                the linear component of the constraint.

                Note
                    lin_expr must not contain duplicate indices.  If lin_expr
                    references a variable more than once, either by index, name,
                    or a combination of index and name, an exception will be
                    raised.

            quad_expr: either a SparseTriple or a list of three lists
                specifying the quadratic component of the constraint.

                Note
                    quad_expr must not contain duplicate indices.  If quad_expr
                    references a matrix entry more than once, either by indices,
                    names, or a combination of indices and names, an exception
                    will be raised.

            sense: either "L", "G", or "E"

            rhs: a float specifying the righthand side of the constraint.

            name: the name of the constraint.

        Returns:
            The index of the added quadratic constraint.

        Raises:
            QiskitOptimizationError: if arguments are not valid.

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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
            name = 'q{}'.format(1 + self.get_num())
        if name in self._index:
            raise QiskitOptimizationError('Duplicate quadratic constraint name: {}'.format(name))
        self._names.append(name)
        self._index.build(self._names)

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
            i_2 = self._varindex(i)
            if i_2 in lin_expr_dict:
                raise QiskitOptimizationError('lin_expr contains duplicate index: {}'.format(i))
            lin_expr_dict[i_2] = val
        self._lin_expr.append(lin_expr_dict)

        # quadratic terms
        quad_matrix = dok_matrix((0, 0))
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
            i_2 = self._varindex(i)
            j_2 = self._varindex(j)
            if i_2 < j_2:
                # to reproduce CPLEX's behavior, swap i_2 and j_2 so that i_2 >= j_2
                i_2, j_2 = j_2, i_2
            if (i_2, j_2) in quad_matrix:
                raise QiskitOptimizationError(
                    'quad_expr contains duplicate index: {} {}'.format(i, j))
            max_ij = max(i_2, j_2)
            if max_ij >= quad_matrix.shape[0]:
                quad_matrix.resize(max_ij + 1, max_ij + 1)
            quad_matrix[i_2, j_2] = val
        self._quad_expr.append(quad_matrix)

        if sense not in ['L', 'G', 'E']:
            raise QiskitOptimizationError('Invalid sense: {}'.format(sense))
        self._senses.append(sense)
        self._rhs.append(rhs)

        return self._index.convert(name)

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

        Example usage:

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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
            self._index = NameIndex()
            return

        keys = self._index.convert(*args)
        if isinstance(keys, int):
            keys = [keys]
        for i in sorted(keys, reverse=True):
            del self._rhs[i]
            del self._senses[i]
            del self._names[i]
            del self._lin_expr[i]
            del self._quad_expr[i]
        self._index.build(self._names)

    def get_rhs(self, *args) -> Union[float, List[float]]:
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

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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

        def _get(i):
            return self._rhs[i]

        if len(args) == 0:
            return copy.deepcopy(self._rhs)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_senses(self, *args) -> Union[str, List[str]]:
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

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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

        def _get(i):
            return self._senses[i]

        if len(args) == 0:
            return copy.deepcopy(self._senses)
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

    def get_linear_num_nonzeros(self, *args) -> Union[int, List[int]]:
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

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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

        def _nonzero(i) -> int:
            tab = self._lin_expr[i]
            return len([0 for v in tab.values() if v != 0.0])

        keys = self._index.convert(*args)
        return self._getter(_nonzero, keys)

    def get_linear_components(self, *args) -> Union[SparsePair, List[SparsePair]]:
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

        Examples:

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
        >>> indices = op.variables.add(
        ...     names=[str(i) for i in range(4)],
        ...     types="B" * 4
        ... )
        >>> [op.quadratic_constraints.add(
        ...      name=str(i),
        ...      lin_expr=[range(i), [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(3)]
        [0, 1, 2]
        >>> op.quadratic_constraints.get_num()
        3
        >>> op.quadratic_constraints.get_linear_components(2)
        SparsePair(ind = [0, 1], val = [1.0, 2.0])
        >>> for row in op.quadratic_constraints.get_linear_components("0", 1):
        ...     print(row)
        SparsePair(ind = [], val = [])
        SparsePair(ind = [0], val = [1.0])
        >>> for row in op.quadratic_constraints.get_linear_components([1, "0"]):
        ...     print(row)
        SparsePair(ind = [0], val = [1.0])
        SparsePair(ind = [], val = [])
        >>> for row in op.quadratic_constraints.get_linear_components():
        ...     print(row)
        SparsePair(ind = [], val = [])
        SparsePair(ind = [0], val = [1.0])
        SparsePair(ind = [0, 1], val = [1.0, 2.0])
        """

        def _linear_component(i) -> SparsePair:
            tab = self._lin_expr[i]
            return SparsePair(ind=list(tab.keys()), val=list(tab.values()))

        keys = self._index.convert(*args)
        return self._getter(_linear_component, keys)

    def get_quad_num_nonzeros(self, *args) -> Union[int, List[int]]:
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

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
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

        def _nonzero(i) -> int:
            return self._quad_expr[i].nnz

        keys = self._index.convert(*args)
        return self._getter(_nonzero, keys)

    def get_quadratic_components(self, *args) -> Union[SparseTriple, List[SparseTriple]]:
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

        >>> from qiskit.optimization import QuadraticProgram
        >>> op = QuadraticProgram()
        >>> indices = op.variables.add(
        ...     names=[str(i) for i in range(4)]
        ... )
        >>> [op.quadratic_constraints.add(
        ...      name="q{0}".format(i),
        ...      quad_expr=[range(i), range(i),
        ...                 [1.0 * (j+1.0) for j in range(i)]])
        ...  for i in range(1, 3)]
        [0, 1]
        >>> op.quadratic_constraints.get_num()
        2
        >>> op.quadratic_constraints.get_quadratic_components(1)
        SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0])
        >>> for quad in op.quadratic_constraints.get_quadratic_components("q1", 1):
        ...     print(quad)
        SparseTriple(ind1 = [0], ind2 = [0], val = [1.0])
        SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0])
        >>> for quad in op.quadratic_constraints.get_quadratic_components(["q2", 0]):
        ...     print(quad)
        SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0])
        SparseTriple(ind1 = [0], ind2 = [0], val = [1.0])
        >>> for quad in op.quadratic_constraints.get_quadratic_components():
        ...     print(quad)
        SparseTriple(ind1 = [0], ind2 = [0], val = [1.0])
        SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [1.0, 2.0])
        """

        def _quadratic_component(k) -> SparseTriple:
            ind1 = []
            ind2 = []
            val = []
            mat = self._quad_expr[k]
            for (i, j), v in mat.items():
                ind1.append(i)
                ind2.append(j)
                val.append(v)
            return SparseTriple(ind1=ind1, ind2=ind2, val=val)

        keys = self._index.convert(*args)
        return self._getter(_quadratic_component, keys)

    def get_names(self, *args) -> Union[str, List[str]]:
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

        >>> from qiskit.optimization.problems import QuadraticProgram
        >>> op = QuadraticProgram()
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

        def _get(i):
            return self._names[i]

        if len(args) == 0:
            return self._names
        keys = self._index.convert(*args)
        return self._getter(_get, keys)

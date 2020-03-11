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


from typing import Callable, Sequence, Union, Any

from qiskit.optimization.utils.helpers import NameIndex
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError


class BaseInterface(object):
    """Common methods for sub-interfaces within Qiskit Optimization."""

    def __init__(self):
        """Creates a new BaseInterface.

        This class is not meant to be instantiated directly nor used
        externally.
        """
        if type(self) == BaseInterface:
            raise TypeError("BaseInterface must be sub-classed")
        self._index = NameIndex()

    def get_indices(self, *args):
        """Converts from names to indices.

        If name is a string, get_indices returns the index of the
        object with that name.  If no such object exists, an
        exception is raised.

        If name is a sequence of strings, get_indices returns a list
        of the indices corresponding to the strings in name.
        Equivalent to map(self.get_indices, name).

        If the subclass does not provide an index function (i.e., the
        interface is not indexed), then a NotImplementedError is raised.

        Example usage:

        >>> from qiskit.optimization.problems import OptimizationProblem
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=["a", "b"])
        >>> c.variables.get_indices("a")
        0
        >>> c.variables.get_indices(["a", "b"])
        [0, 1]
        """
        return self._index.convert(*args)

    @staticmethod
    def _setter(setfunc: Callable[[int, Any], None], *args):
        # check for all elements in args whether they are types
        if len(args) == 1 and all(isinstance(el, Sequence) and len(el) == 2 for el in args[0]):
            for el in args[0]:
                setfunc(*el)
        elif len(args) == 2:
            setfunc(*args)
        else:
            raise QiskitOptimizationError("Invalid arguments: {}".format(args))

    @staticmethod
    def _getter(getfunc: Callable[[int], Any], *args):
        # `args` should be already converted into `int` by `get_indices`.
        if len(args) == 0:
            raise QiskitOptimizationError('Empty arguments should be handled in the caller')
        if len(args) == 1:
            if isinstance(args[0], Sequence):
                args = args[0]
            else:
                return getfunc(args[0])
        return [getfunc(k) for k in args]

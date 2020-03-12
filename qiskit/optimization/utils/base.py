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


from typing import Callable, Sequence, Union, Any, List

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

    def get_indices(self, *name) -> Union[int, List[int]]:
        """Converts from names to indices.

        If name is a string, get_indices returns the index of the
        object with that name.  If no such object exists, an
        exception is raised.

        If name is a sequence of strings, get_indices returns a list
        of the indices corresponding to the strings in name.
        Equivalent to map(self.get_indices, name).

        See `NameIndex.convert` for details.

        If the subclass does not provide an index function (i.e., the
        interface is not indexed), then a NotImplementedError is raised.

        Example usage:

        >>> from qiskit.optimization import OptimizationProblem
        >>> op = OptimizationProblem()
        >>> indices = op.variables.add(names=["a", "b"])
        >>> op.variables.get_indices("a")
        0
        >>> op.variables.get_indices(["a", "b"])
        [0, 1]
        """
        return self._index.convert(*name)

    @staticmethod
    def _setter(setfunc: Callable[[Union[str, int], Any], None], *args) -> None:
        """A generic setter method

        Args:
            setfunc(index, val): A setter function of two parameters: `index` and `val`.
                Since `index` can be a string, users need to convert it into an appropriate index
                by applying `NameIndex.convert`.
            *args: A pair of index and value or a list of pairs of index and value.
                `setfunc` is invoked with `args`.

        Returns:
            None
        """
        # check for all elements in args whether they are types
        if len(args) == 1 and \
                all(isinstance(pair, Sequence) and len(pair) == 2 for pair in args[0]):
            for pair in args[0]:
                setfunc(*pair)
        elif len(args) == 2:
            setfunc(*args)
        else:
            raise QiskitOptimizationError("Invalid arguments: {}".format(args))

    @staticmethod
    def _getter(getfunc: Callable[[int], Any], *args) -> Any:
        """A generic getter method

        Args:
            getfunc(index): A getter function with an argument `index`.
                `index` should be already converted by `NameIndex.convert`.
            *args: A single index or a list of indices. `getfunc` is invoked with args.

        Returns: if `args` is a single index, this returns a single value genereted by `getfunc`.
            If `args` is a list of indices, this returns a list of values.
        """
        if len(args) == 0:
            raise QiskitOptimizationError('Empty arguments should be handled in the caller')
        if len(args) == 1:
            if isinstance(args[0], Sequence):
                args = args[0]
            else:
                return getfunc(args[0])
        return [getfunc(k) for k in args]

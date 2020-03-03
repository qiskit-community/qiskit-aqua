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


from qiskit.optimization.utils.helpers import convert, _defaultgetindexfunc


class BaseInterface(object):
    """Common methods for sub-interfaces within Qiskit Optimization."""

    def __init__(self):
        """Creates a new BaseInterface.

        This class is not meant to be instantiated directly nor used
        externally.
        """
        if type(self) == BaseInterface:
            raise TypeError("BaseInterface must be sub-classed")

    def _conv(self, name, cache=None):
        """Converts from names to indices as necessary."""
        return convert(name, self._get_index, cache)

    @staticmethod
    def _add_iter(getnumfun, addfun, *args, **kwargs):
        """non-public"""
        old = getnumfun()
        addfun(*args, **kwargs)
        return range(old, getnumfun())

    @staticmethod
    def _add_single(getnumfun, addfun, *args, **kwargs):
        """non-public"""
        addfun(*args, **kwargs)
        return getnumfun() - 1  # minus one for zero-based indices

    def _get_index(self, name):
        return _defaultgetindexfunc(name)

    def get_indices(self, name):
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

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=["a", "b"])
        >>> c.variables.get_indices("a")
        0
        >>> c.variables.get_indices(["a", "b"])
        [0, 1]
        """
        if _defaultgetindexfunc is None:
            raise NotImplementedError("This is not an indexed interface")
        if isinstance(name, str):
            return self._get_index(name)
        else:
            return [self._get_index(x) for x in name]

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


from typing import Union, List, Sequence

from qiskit.optimization.utils import QiskitOptimizationError


class NameIndex(object):
    """Convert a string name into an integer index.
    This is used for the implementation of `BaseInterface.get_indices`.
    """

    def __init__(self):
        """Initialize a dictionary of name and index"""
        self._dict = {}

    def __contains__(self, name: str) -> bool:
        """Check a name is registered or not.

        Args:
            name: a string name

        Returns:
            This returns True if the name has been registered. Otherwise it returns False.

        """
        return name in self._dict

    def build(self, names: List[str]) -> None:
        """Build a dictionary from scratch.

        Args:
            names: a list of names
        """
        self._dict = {e: i for i, e in enumerate(names)}

    def _convert_one(self, arg: Union[str, int]) -> int:
        if isinstance(arg, int):
            return arg
        if not isinstance(arg, str):
            raise QiskitOptimizationError('Invalid argument" {}'.format(arg))
        if arg not in self._dict:
            self._dict[arg] = len(self._dict)
        return self._dict[arg]

    def convert(self, *args) -> Union[int, List[int]]:
        """Convert a set of names into a set of indices.
        There are three types of arguments.

        - `convert()` returns all indices.

        - `convert(Union[str, int])` return an index corresponding to the argument.
            If the argument is already integer, this returns the same integer value.

        - `convert(List[Union[str, int]])` returns a list of indices

        - `convert(begin, end)` return a list of indices in a range starting from `begin` to `end`,
            which includes both `begin` and `end`.
            Note that it behaves similar to `range(begin, end+1)`
        """
        if len(args) == 0:
            return list(self._dict.values())
        elif len(args) == 1:
            a_0 = args[0]
            if isinstance(a_0, (int, str)):
                return self._convert_one(a_0)
            elif isinstance(a_0, Sequence):
                return [self._convert_one(e) for e in a_0]
            else:
                raise QiskitOptimizationError('Invalid argument: {}'.format(args))
        elif len(args) == 2:
            begin = self._convert_one(args[0])
            end = self._convert_one(args[1]) + 1
            return list(range(begin, end))
        else:
            raise QiskitOptimizationError('Invalid arguments: {}'.format(args))


def init_list_args(*args):
    """Initialize default arguments with empty lists if necessary."""
    return tuple([] if a is None else a for a in args)

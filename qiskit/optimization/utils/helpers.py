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


from typing import Union, List, Dict, Sequence

from qiskit.optimization.utils import QiskitOptimizationError


class NameIndex:
    def __init__(self):
        self._dict = {}

    def to_dict(self) -> Dict[str, int]:
        return self._dict

    def __contains__(self, item: str) -> bool:
        return item in self._dict

    def build(self, names: List[str]):
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
        if len(args) == 0:
            return list(self._dict.values())
        elif len(args) == 1:
            a0 = args[0]
            if isinstance(a0, (int, str)):
                return self._convert_one(a0)
            elif isinstance(a0, Sequence):
                return [self._convert_one(e) for e in a0]
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

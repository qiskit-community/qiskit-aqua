# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Trotterization Algorithm Base """

import logging
from abc import abstractmethod, ABC

from ...operator_base import OperatorBase

# TODO centralize handling of commuting groups

logger = logging.getLogger(__name__)


class TrotterizationBase(ABC):
    """ A base for Trotterization methods, algorithms for approximating exponentiations of
    operator sums by compositions of exponentiations.
    """

    def __init__(self, reps: int = 1) -> None:
        self._reps = reps

    @property
    def reps(self) -> int:
        """ returns reps """
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        self._reps = reps

    @abstractmethod
    def trotterize(self, op_sum: OperatorBase) -> OperatorBase:
        """ trotterize """
        raise NotImplementedError

    # TODO @abstractmethod - trotter_error_bound

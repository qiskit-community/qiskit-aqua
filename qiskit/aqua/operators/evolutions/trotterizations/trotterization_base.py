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

""" Expectation Algorithm Base """

import logging
from abc import abstractmethod

# TODO centralize handling of commuting groups

logger = logging.getLogger(__name__)


class TrotterizationBase():
    """ A base for Trotterization methods to allow for user-specified trotterization. """

    @staticmethod
    def factory(mode, reps=1):
        if mode not in ['trotter', 'suzuki', 'qdrift']:
            raise ValueError('Trotter mode {} not supported'.format(mode))
        if mode == 'trotter':
            from .trotter import Trotter
            return Trotter(reps=reps)
        if mode == 'suzuki':
            from .suzuki import Suzuki
            return Suzuki(reps=reps)
        if mode == 'qdrift':
            from .qdrift import QDrift
            return QDrift(reps=reps)

    def __init__(self, reps=1):
        self._reps = reps

    @property
    def reps(self):
        return self._reps

    @reps.setter
    def reps(self, order):
        self._reps = order

    @abstractmethod
    def trotterize(self, op_sum):
        raise NotImplementedError

    # TODO @abstractmethod - trotter_error_bound

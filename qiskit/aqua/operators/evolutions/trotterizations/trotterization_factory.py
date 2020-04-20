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

""" Trotterization Algorithm Factory """

import logging

from .trotterization_base import TrotterizationBase
from .trotter import Trotter
from .suzuki import Suzuki
from .qdrift import QDrift

logger = logging.getLogger(__name__)


class TrotterizationFactory():
    """ A factory for creating Trotterization algorithms. """

    @staticmethod
    # pylint: disable=inconsistent-return-statements
    def build(mode: str,
              reps: int = 1) -> TrotterizationBase:
        """ Factory method for constructing Trotterization algorithms. """
        if mode not in ['trotter', 'suzuki', 'qdrift']:
            raise ValueError('Trotter mode {} not supported'.format(mode))

        # pylint: disable=cyclic-import
        if mode == 'trotter':
            return Trotter(reps=reps)

        if mode == 'suzuki':
            return Suzuki(reps=reps)

        if mode == 'qdrift':
            return QDrift(reps=reps)

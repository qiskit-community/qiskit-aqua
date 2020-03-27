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

""" Projector Overlap Expectation Value algorithm.  """

import logging
from typing import Union
import numpy as np

from qiskit.providers import BaseBackend

from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase

logger = logging.getLogger(__name__)


class ProjectorOverlap(ExpectationBase):
    """ Projector Overlap Expectation Value algorithm.  """

    def __init__(self,
                 operator: OperatorBase = None,
                 state: OperatorBase = None,
                 backend: BaseBackend = None) -> None:
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self.backend = backend

    def compute_expectation(self,
                            state: OperatorBase = None,
                            params: dict = None) -> Union[list, float, complex, np.ndarray]:
        """ compute expectation """
        raise NotImplementedError

    def compute_standard_deviation(self,
                                   state: OperatorBase = None,
                                   params: dict = None) -> Union[list, float, complex, np.ndarray]:
        """ compute standard deviation """
        raise NotImplementedError

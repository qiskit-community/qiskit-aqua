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
from typing import Union
from abc import abstractmethod
import numpy as np

from ..operator_base import OperatorBase
from ..converters import ConverterBase

logger = logging.getLogger(__name__)


class ExpectationBase(ConverterBase):
    """ A base for Expectation Value algorithms. An expectation value algorithm
    takes an operator Observable,
    a backend, and a state distribution function, and computes the expected value
    of that observable over the
    distribution.

    """

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept an Operator and return a new Operator with the measurements replaced by
        alternate methods to compute the expectation value. """
        raise NotImplementedError

    @abstractmethod
    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float, complex, np.ndarray]:
        """ compute variance """
        raise NotImplementedError

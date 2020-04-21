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

from qiskit.providers import BaseBackend
from ..operator_base import OperatorBase
from ..converters import ConverterBase
from ..circuit_samplers import CircuitSamplerFactory

logger = logging.getLogger(__name__)


class ExpectationBase(ConverterBase):
    """ A base for Expectation Value algorithms. An expectation value algorithm
    takes an operator Observable,
    a backend, and a state distribution function, and computes the expected value
    of that observable over the
    distribution.

    # TODO make into QuantumAlgorithm to make backend business consistent?

    """

    def __init__(self) -> None:
        self._circuit_sampler = None

    @property
    def backend(self) -> BaseBackend:
        """ returns backend """
        return self._circuit_sampler.backend

    @backend.setter
    def backend(self, backend: BaseBackend) -> None:
        if backend is not None:
            self._circuit_sampler = CircuitSamplerFactory.build(backend=backend)

    # TODO change VQE to rely on this instead of compute_expectation
    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept an Operator and return a new Operator with the measurements replaced by
        alternate methods to compute the expectation value. """
        raise NotImplementedError

    @property
    @abstractmethod
    def operator(self) -> OperatorBase:
        """ returns operator """
        raise NotImplementedError

    @abstractmethod
    def compute_expectation(self,
                            state: OperatorBase = None,
                            params: dict = None) -> Union[list, float, complex, np.ndarray]:
        """ compute expectation """
        raise NotImplementedError

    @abstractmethod
    def compute_standard_deviation(self,
                                   state: OperatorBase = None,
                                   params: dict = None) -> Union[list, float, complex, np.ndarray]:
        """ compute variance """
        raise NotImplementedError

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
import numpy as np
from abc import abstractmethod

from ..converters import ConverterBase

from qiskit.aqua.utils.backend_utils import (is_ibmq_provider,
                                             is_local_backend,
                                             has_aer,
                                             is_statevector_backend,
                                             is_aer_qasm)
from qiskit.aqua import QuantumInstance

logger = logging.getLogger(__name__)


class CircuitSampler(ConverterBase):
    """ A base for Expectation Value algorithms. An expectation
    value algorithm takes an operator Observable,
    a backend, and a state distribution function,
    and computes the expected value of that observable over the
    distribution.

    """

    @staticmethod
    def factory(backend=None):
        """ A factory method to produce the correct type of CircuitSampler
        subclass based on the primitive passed in."""

        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend

        if is_local_backend(backend_to_check):
            from . import LocalSimulatorSampler
            return LocalSimulatorSampler(backend=backend,
                                         statevector=is_statevector_backend(backend_to_check),
                                         snapshot=is_aer_qasm(backend_to_check))

        if is_ibmq_provider(backend_to_check):
            from . import IBMQSampler
            return IBMQSampler(backend=backend)

    @abstractmethod
    def convert(self, operator):
        """ Accept the Operator and return the converted Operator """
        raise NotImplementedError

    @abstractmethod
    def sample_circuits(self, op_circuits):
        """ Accept a list of op_circuits and return a list of count dictionaries for each."""
        raise NotImplementedError

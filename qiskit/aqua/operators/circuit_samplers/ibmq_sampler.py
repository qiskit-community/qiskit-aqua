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

""" Expectation Algorithm Base """

import logging
import numpy as np
from abc import abstractmethod

from qiskit import BasicAer

from . import CircuitSampler
from qiskit.aqua import AquaError, QuantumAlgorithm, QuantumInstance

logger = logging.getLogger(__name__)


class IBMQSampler(CircuitSampler):
    """ A sampler for local Quantum simulator backends.

    """

    def __init__(self, backend, hw_backend_to_emulate=None, kwargs={}):
        """
        Args:
            backend():
            hw_backend_to_emulate():
        """
        self._backend = backend
        if has_aer and 'noise_model' not in kwargs:
            from qiskit.providers.aer.noise import NoiseModel
            kwargs['noise_model'] = NoiseModel.from_backend(hw_backend_to_emulate)
        self._qi = QuantumInstance(backend=backend, **kwargs)

    def convert(self, operator):
        reduced_op = operator.reduce()

    def sample_circuits(self, op_circuits):
        """
        Args:
            op_circuits(list): The list of circuits to sample
        """
        pass

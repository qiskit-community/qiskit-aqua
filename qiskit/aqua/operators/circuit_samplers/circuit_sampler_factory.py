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

""" Circuit Sampler Factory """

from typing import Union
import logging

from qiskit.providers import BaseBackend

from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils.backend_utils import (is_ibmq_provider,
                                             is_local_backend,
                                             is_statevector_backend,
                                             is_aer_qasm)
from .circuit_sampler_base import CircuitSamplerBase
from .local_simulator_sampler import LocalSimulatorSampler
from .ibmq_sampler import IBMQSampler

logger = logging.getLogger(__name__)


class CircuitSamplerFactory():
    """ A factory for convenient construction of Circuit Samplers.
    """

    @staticmethod
    # pylint: disable=inconsistent-return-statements
    def build(backend: Union[BaseBackend, QuantumInstance]) -> CircuitSamplerBase:
        """ A factory method to produce the correct type of CircuitSamplerBase
        subclass based on the primitive passed in."""

        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend
        # pylint: disable=cyclic-import,import-outside-toplevel
        if is_local_backend(backend_to_check):
            return LocalSimulatorSampler(backend=backend,
                                         statevector=is_statevector_backend(backend_to_check),
                                         snapshot=is_aer_qasm(backend_to_check))

        if is_ibmq_provider(backend_to_check):
            return IBMQSampler(backend=backend)

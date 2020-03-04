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
from qiskit.aqua.operators.circuit_samplers import CircuitSampler

from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_qasm,
                                             has_aer)

logger = logging.getLogger(__name__)


class ExpectationBase():
    """ A base for Expectation Value algorithms. An expectation value algorithm takes an operator Observable,
    a backend, and a state distribution function, and computes the expected value of that observable over the
    distribution.

    """

    def __init__(self):
        self._circuit_sampler = None

    def set_backend(self, backend=None):
        self._circuit_sampler = CircuitSampler.factory(backend=backend)

    @staticmethod
    def factory(operator, backend=None, state=None):
        """
        Args:

        """
        primitives = operator.get_primtives()
        if primitives == {'Pauli'}:

            if backend is None:
                # If user has Aer but didn't specify a backend, use the Aer fast expectation
                if has_aer():
                    from qiskit import Aer
                    backend = Aer.get_backend('qasm_simulator')
                # If user doesn't have Aer, use statevector_simulator for < 16 qubits, and qasm with warning for more.
                else:
                    if operator.num_qubits <= 16:
                        backend = BasicAer.get_backend('statevector_simulator')
                    else:
                        logging.warning('{0} qubits is a very large expectation value. Consider installing Aer to use '
                                        'Aer\'s fast expectation, which will perform better here. We\'ll use '
                                        'the BasicAer qasm backend for this expectation to avoid having to '
                                        'construct the {1}x{1} operator matrix.'.format(operator.num_qubits,
                                                                                        2**operator.num_qubits))
                        backend = BasicAer.get_backend('qasm_simulator')

            # If the user specified Aer qasm backend and is using a Pauli operator, use the Aer fast expectation
            if is_aer_qasm(backend):
                from .aer_pauli_expectation import AerPauliExpectation
                return AerPauliExpectation(operator=operator, backend=backend, state=state)

            # If the user specified a statevector backend (either Aer or BasicAer), use a converter to produce a
            # Matrix operator and compute using matmul
            elif is_statevector_backend(backend):
                from .matrix_expectation import MatrixExpectation
                if operator.num_qubits >= 16:
                    logging.warning('Note: Using a statevector_simulator with {} qubits can be very expensive. '
                                    'Consider using the Aer qasm_simulator instead to take advantage of Aer\'s '
                                    'built-in fast Pauli Expectation'.format(operator.num_qubits))
                # TODO do this properly with converters
                return MatrixExpectation(operator=operator, backend=backend, state=state)

            # All other backends, including IBMQ, BasicAer QASM, go here.
            else:
                from .pauli_expectation import PauliExpectation
                return PauliExpectation(operator=operator, backend=backend, state=state)

        elif primitives == {'Matrix'}:
            from .matrix_expectation import MatrixExpectation
            return MatrixExpectation(operator=operator, backend=backend, state=state)

        elif primitives == {'Instruction'}:
            from .projector_overlap import ProjectorOverlap
            return ProjectorOverlap(operator=operator, backend=backend, state=state)

        else:
            raise ValueError('Expectations of Mixed Operators not yet supported.')

    @abstractmethod
    def compute_expectation_for_primitives(self, state=None, primitives=None):
        raise NotImplementedError

    @abstractmethod
    def compute_variance(self, state=None):
        raise NotImplementedError

    def compute_expectation(self, state=None):
        pass

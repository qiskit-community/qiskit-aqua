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

from qiskit import BasicAer
from qiskit.providers import BaseBackend
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_qasm,
                                             has_aer)
from qiskit.aqua import QuantumInstance
from ..operator_base import OperatorBase
from ..circuit_samplers import CircuitSampler

logger = logging.getLogger(__name__)


class ExpectationBase:
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
            self._circuit_sampler = CircuitSampler.factory(backend=backend)

    @staticmethod
    def factory(operator: OperatorBase,
                backend: BaseBackend = None,
                state: OperatorBase = None):
        """
        Args:
        Returns:
            ExpectationBase: derived class
        Raises:
            ValueError: Expectations of Mixed Operators not yet supported.
        """
        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend

        # pylint: disable=cyclic-import,import-outside-toplevel
        # TODO remove state from factory and inits?
        primitives = operator.get_primitives()
        if primitives == {'Pauli'}:

            if backend_to_check is None:
                # If user has Aer but didn't specify a backend, use the Aer fast expectation
                if has_aer():
                    from qiskit import Aer
                    backend_to_check = Aer.get_backend('qasm_simulator')
                # If user doesn't have Aer, use statevector_simulator
                # for < 16 qubits, and qasm with warning for more.
                else:
                    if operator.num_qubits <= 16:
                        backend_to_check = BasicAer.get_backend('statevector_simulator')
                    else:
                        logging.warning(
                            '%d qubits is a very large expectation value. '
                            'Consider installing Aer to use '
                            'Aer\'s fast expectation, which will perform better here. We\'ll use '
                            'the BasicAer qasm backend for this expectation to avoid having to '
                            'construct the %dx%d operator matrix.',
                            operator.num_qubits,
                            2 ** operator.num_qubits,
                            2 ** operator.num_qubits)
                        backend_to_check = BasicAer.get_backend('qasm_simulator')

            # If the user specified Aer qasm backend and is using a
            # Pauli operator, use the Aer fast expectation
            if is_aer_qasm(backend_to_check):
                from .aer_pauli_expectation import AerPauliExpectation
                return AerPauliExpectation(operator=operator, backend=backend, state=state)

            # If the user specified a statevector backend (either Aer or BasicAer),
            # use a converter to produce a
            # Matrix operator and compute using matmul
            elif is_statevector_backend(backend_to_check):
                from .matrix_expectation import MatrixExpectation
                if operator.num_qubits >= 16:
                    logging.warning(
                        'Note: Using a statevector_simulator with %d qubits can be very expensive. '
                        'Consider using the Aer qasm_simulator instead to take advantage of Aer\'s '
                        'built-in fast Pauli Expectation', operator.num_qubits)
                # TODO do this properly with converters
                return MatrixExpectation(operator=operator, backend=backend, state=state)

            # All other backends, including IBMQ, BasicAer QASM, go here.
            else:
                from .pauli_expectation import PauliExpectation
                return PauliExpectation(operator=operator, backend=backend, state=state)

        elif primitives == {'Matrix'}:
            from .matrix_expectation import MatrixExpectation
            return MatrixExpectation(operator=operator, backend=backend, state=state)

        elif primitives == {'QuantumCircuit'}:
            from .projector_overlap import ProjectorOverlap
            return ProjectorOverlap(operator=operator, backend=backend, state=state)

        else:
            raise ValueError('Expectations of Mixed Operators not yet supported.')

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

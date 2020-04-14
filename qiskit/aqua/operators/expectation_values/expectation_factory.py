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

from typing import Union, Optional
import logging

from qiskit import BasicAer
from qiskit.providers import BaseBackend
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_qasm,
                                             has_aer)
from qiskit.aqua import QuantumInstance

from .expectation_base import ExpectationBase
from .aer_pauli_expectation import AerPauliExpectation
from .pauli_expectation import PauliExpectation
from .matrix_expectation import MatrixExpectation
from ..operator_base import OperatorBase

logger = logging.getLogger(__name__)


class ExpectationFactory:
    """ A factory for creating ExpectationBase algorithms given an operator, backend, and state.

    """

    @staticmethod
    def build(operator: OperatorBase,
              backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
              state: Optional[OperatorBase] = None) -> ExpectationBase:
        """
        Args:
        Returns:
            ExpectationBase: derived class
        Raises:
            ValueError: Expectations of Mixed Operators not yet supported.
        """
        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend

        # pylint: disable=cyclic-import,import-outside-toplevel
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
                return AerPauliExpectation(operator=operator, backend=backend, state=state)

            # If the user specified a statevector backend (either Aer or BasicAer),
            # use a converter to produce a
            # Matrix operator and compute using matmul
            elif is_statevector_backend(backend_to_check):
                if operator.num_qubits >= 16:
                    logging.warning(
                        'Note: Using a statevector_simulator with %d qubits can be very expensive. '
                        'Consider using the Aer qasm_simulator instead to take advantage of Aer\'s '
                        'built-in fast Pauli Expectation', operator.num_qubits)
                # TODO do this properly with converters
                return MatrixExpectation(operator=operator, backend=backend, state=state)

            # All other backends, including IBMQ, BasicAer QASM, go here.
            else:
                return PauliExpectation(operator=operator, backend=backend, state=state)

        elif primitives == {'Matrix'}:
            return MatrixExpectation(operator=operator, backend=backend, state=state)

        else:
            raise ValueError('Expectations of Mixed Operators not yet supported.')

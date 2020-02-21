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

from .expectation_base import ExpectationBase

from qiskit.aqua.operators import OpVec, OpPrimitive
from qiskit.aqua.operators.converters import PauliChangeOfBasis

logger = logging.getLogger(__name__)


class PauliExpectation(ExpectationBase):
    """ An Expectation Value algorithm for taking expectations of quantum states specified by circuits over
    observables specified by Pauli Operators. Flow:

    """

    def __init__(self, operator=None, state=None, backend=None):
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self.set_backend(backend)
        self._converted_operator = None
        self._reduced_expect_op = None

    # TODO setters which wipe state

    def compute_expectation(self, state=None):
        if state or not self._reduced_expect_op:
            self._reduced_expect_op = self.expectation_op(state=state)
            # TODO to_quantum_runnable converter?

        if 'Instruction' in self._reduced_expect_op.get_primtives():
            # TODO check if params have been sufficiently provided.
            if self._circuit_sampler:
                measured_op = self._circuit_sampler.run_circuits(self._reduced_expect_op)
                return measured_op.eval()
            else:
                raise ValueError('Unable to compute expectation of functions containing circuits without a backend '
                                 'set. Set a backend for the Expectation algorithm to compute the expectation, '
                                 'or convert Instructions to other types which do not require a backend.')
        else:
            return self._reduced_expect_op.eval()

    def expectation_op(self, state=None):
        # TODO allow user to set state in constructor and then only pass params to execute.
        state = state or self._state

        if not self._converted_operator:
            # Construct measurement from operator
            meas = self._operator.as_measurement()
            # Convert the measurement into a classical basis (PauliChangeOfBasis chooses this basis by default).
            self._converted_operator = PauliChangeOfBasis().convert(meas)

        expec_op = self._converted_operator.compose(state)
        return expec_op.reduce()

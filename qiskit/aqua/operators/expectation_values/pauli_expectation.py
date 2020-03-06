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

from qiskit.aqua.operators import OpVec, OpPrimitive, StateFn, OpComposition
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
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    # TODO setters which wipe state

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator
        self._converted_operator = None
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    @property
    def quantum_instance(self):
        return self._circuit_sampler.quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance):
        self._circuit_sampler.quantum_instance = quantum_instance

    def expectation_op(self, state=None):
        state = state or self._state

        if not self._converted_operator:
            # Construct measurement from operator
            meas = StateFn(self._operator, is_measurement=True)
            # Convert the measurement into a classical basis (PauliChangeOfBasis chooses this basis by default).
            self._converted_operator = PauliChangeOfBasis().convert(meas)
            # TODO self._converted_operator = PauliExpectation.group_equal_measurements(self._converted_operator)

        expec_op = self._converted_operator.compose(state)
        return expec_op.reduce()

    def compute_expectation(self, state=None, params=None):
        # Wipes caches in setter
        if state and not state == self.state:
            self.state = state

        if not self._reduced_meas_op:
            self._reduced_meas_op = self.expectation_op(state=state)

        if 'Instruction' in self._reduced_meas_op.get_primitives():
            # TODO check if params have been sufficiently provided.
            if self._circuit_sampler:
                self._sampled_meas_op = self._circuit_sampler.convert(self._reduced_meas_op, params=params)
                return self._sampled_meas_op.eval()
            else:
                raise ValueError('Unable to compute expectation of functions containing circuits without a backend '
                                 'set. Set a backend for the Expectation algorithm to compute the expectation, '
                                 'or convert Instructions to other types which do not require a backend.')
        else:
            return self._reduced_meas_op.eval()

    def compute_standard_deviation(self, state=None, params=None):
        state = state or self.state
        if self._sampled_meas_op is None:
            self.compute_expectation(state=state, params=params)

        def sum_variance(operator):
            if isinstance(operator, OpComposition):
                sfdict = operator.oplist[1]
                measurement = operator.oplist[0]
                average = measurement.eval(sfdict)
                variance = sum([(v * (measurement.eval(b) - average))**2
                                for (b, v) in sfdict.primitive.items()])
                return (operator.coeff * variance)**.5
            elif isinstance(operator, OpVec):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])

        return sum_variance(self._sampled_meas_op)

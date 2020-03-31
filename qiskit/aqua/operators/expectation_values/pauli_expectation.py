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

""" Expectation Algorithm for Pauli-basis observables by changing to diagonal basis and
estimating average by sampling. """

import logging
from typing import Union
import numpy as np

from qiskit.providers import BaseBackend

from qiskit.aqua import QuantumInstance
from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..combo_operators import ListOp, ComposedOp
from ..state_functions import StateFn
from ..converters import PauliBasisChange, AbelianGrouper

logger = logging.getLogger(__name__)


class PauliExpectation(ExpectationBase):
    """ An Expectation Value algorithm for taking expectations of quantum states
    specified by circuits over observables specified by Pauli Operators.

    Observables are changed to diagonal basis by clifford circuits and average is estimated by
    sampling measurements in the Z-basis.

    """

    def __init__(self,
                 operator: OperatorBase = None,
                 state: OperatorBase = None,
                 backend: BaseBackend = None,
                 group_paulis: bool = True) -> None:
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self.backend = backend
        self._grouper = AbelianGrouper() if group_paulis else None
        self._converted_operator = None
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    # TODO setters which wipe state

    @property
    def operator(self) -> OperatorBase:
        return self._operator

    @operator.setter
    def operator(self, operator: OperatorBase) -> None:
        self._operator = operator
        self._converted_operator = None
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    @property
    def state(self) -> OperatorBase:
        """ returns state """
        return self._state

    @state.setter
    def state(self, state: OperatorBase) -> None:
        self._state = state
        self._reduced_meas_op = None
        self._sampled_meas_op = None

    @property
    def quantum_instance(self) -> QuantumInstance:
        """ returns quantum instance """
        return self._circuit_sampler.quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance) -> None:
        self._circuit_sampler.quantum_instance = quantum_instance

    def expectation_op(self, state: OperatorBase = None) -> OperatorBase:
        """ expectation op """
        state = state or self._state

        if not self._converted_operator:
            # Construct measurement from operator
            if self._grouper and isinstance(self._operator, ListOp):
                grouped = self._grouper.convert(self.operator)
                meas = StateFn(grouped, is_measurement=True)
            else:
                meas = StateFn(self._operator, is_measurement=True)
            # Convert the measurement into a classical
            # basis (PauliBasisChange chooses this basis by default).
            cob = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            self._converted_operator = cob.convert(meas)
            # TODO self._converted_operator =
            #  PauliExpectation.group_equal_measurements(self._converted_operator)

        expec_op = self._converted_operator.compose(state)
        return expec_op.reduce()

    def compute_expectation(self,
                            state: OperatorBase = None,
                            params: dict = None) -> Union[list, float, complex, np.ndarray]:
        # Wipes caches in setter
        if state and not state == self.state:
            self.state = state

        if not self._reduced_meas_op:
            self._reduced_meas_op = self.expectation_op(state=state)

        if 'QuantumCircuit' in self._reduced_meas_op.get_primitives():
            # TODO check if params have been sufficiently provided.
            if self._circuit_sampler:
                self._sampled_meas_op = self._circuit_sampler.convert(self._reduced_meas_op,
                                                                      params=params)
                return self._sampled_meas_op.eval()
            else:
                raise ValueError(
                    'Unable to compute expectation of functions containing '
                    'circuits without a backend set. Set a backend for the Expectation '
                    'algorithm to compute the expectation, or convert Instructions to '
                    'other types which do not require a backend.')
        else:
            return self._reduced_meas_op.eval()

    # pylint: disable=inconsistent-return-statements
    def compute_standard_deviation(self,
                                   state: OperatorBase = None,
                                   params: dict = None) -> Union[list, float, complex, np.ndarray]:
        """ compute standard deviation

        TODO Break out into two things - Standard deviation of distribution over observable (mostly
        unchanged with increasing shots), and error of ExpectationValue estimator (decreases with
        increasing shots)
        """
        state = state or self.state
        if self._sampled_meas_op is None:
            self.compute_expectation(state=state, params=params)

        def sum_variance(operator):
            if isinstance(operator, ComposedOp):
                sfdict = operator.oplist[1]
                measurement = operator.oplist[0]
                average = measurement.eval(sfdict)
                variance = sum([(v * (measurement.eval(b) - average))**2
                                for (b, v) in sfdict.primitive.items()])
                return (operator.coeff * variance)**.5
            elif isinstance(operator, ListOp):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])

        return sum_variance(self._sampled_meas_op)

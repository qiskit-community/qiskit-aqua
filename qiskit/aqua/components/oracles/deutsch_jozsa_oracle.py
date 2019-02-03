# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import logging
import math
import numpy
import operator

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import DNF
from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


class DeutschJozsaOracle(Oracle):

    CONFIGURATION = {
        'name': 'DeutschJozsaOracle',
        'description': 'Deutsch Jozsa Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dj_oracle_schema',
            'type': 'object',
            'properties': {
                'bitmap': {
                    "type": ["object", "null"],
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, bitmap, mct_mode='basic'):
        self.validate(locals())
        super().__init__()
        self._mct_mode = mct_mode

        # checks that the input bitstring length is a power of two
        nbits = math.log(len(bitmap), 2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AquaError('Input not the right length')
        self._nbits = int(nbits)

        # checks the input bitstring represents a constant or balanced function
        function = False
        self._bitsum = sum([int(bit) for bit in bitmap.values()])

        if self._bitsum == 0 or self._bitsum == 2 ** self._nbits:
            self._function = "constant"
            function = True
        elif self._bitsum == 2 ** (self._nbits - 1):
            self._function = "balanced"
            function = True
        if not function:
            raise AquaError(
                'Input is not a balanced or constant function')

        def _(bbs):
            return [i[-1] if i[0] == '1' else -i[-1] for i in list(zip(bbs, list(range(1, len(bbs) + 1))))]

        dnf_expr = [_(i) for i in bitmap if bitmap[i] == '1']
        if dnf_expr:
            self._dnf = DNF(dnf_expr)
            self._dnf.construct_circuit()
            self._variable_register = self._dnf.qr_variable
            self._outcome_register = self._dnf.qr_outcome
            self._ancillary_register = self._dnf.qr_ancilla
        else:
            self._dnf = None
            self._variable_register = QuantumRegister(int(nbits), name='v')
            self._outcome_register = QuantumRegister(1, name='o')
            self._ancillary_register = None

    @property
    def variable_register(self):
        return self._variable_register

    @property
    def ancillary_register(self):
        return self._ancillary_register

    @property
    def outcome_register(self):
        return self._outcome_register

    def construct_circuit(self):
        if self._dnf:
            return self._dnf.construct_circuit(mct_mode=self._mct_mode)
        else:
            return QuantumCircuit(self._variable_register)

    def evaluate_classically(self, assignment):
        return self._function == assignment

    def interpret_measurement(self, measurement, *args, **kwargs):
        top_measurement = max(
            measurement.items(), key=operator.itemgetter(1))[0]
        top_measurement = int(top_measurement)
        if top_measurement == 0:
            return "constant"
        else:
            return "balanced"

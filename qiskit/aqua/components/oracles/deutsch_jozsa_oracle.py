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
import operator

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import ESOP
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
                    "type": ["object"],
                },
                'mct_mode': {
                    'type': 'string',
                    'default': 'basic',
                    'oneOf': [
                        {'enum': [
                            'basic',
                            'advanced'
                        ]}
                    ]
                },
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
            raise AquaError('Length of input map must be a power of 2.')
        nbits = int(nbits)

        # checks the input bitstring represents a constant or balanced function
        bitsum = sum([int(bit) for bit in bitmap.values()])

        if bitsum == 0 or bitsum == 2 ** nbits:
            pass  # constant
        elif bitsum == 2 ** (nbits - 1):
            pass  # balanced
        else:
            raise AquaError('Input is not a balanced or constant function.')

        def _(bbs):
            return [i[-1] if i[0] == '1' else -i[-1] for i in list(zip(bbs, list(range(1, len(bbs) + 1))))]

        esop_expr = [_(i) for i in bitmap if bitmap[i] == '1']
        if esop_expr:
            self._esop = ESOP(esop_expr)
            self._esop.construct_circuit()
            self._variable_register = self._esop.qr_variable
            self._outcome_register = self._esop.qr_outcome
            self._ancillary_register = self._esop.qr_ancilla
        else:
            self._esop = None
            self._variable_register = QuantumRegister(nbits, name='v')
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
        if self._esop:
            return self._esop.construct_circuit(mct_mode=self._mct_mode)
        else:
            return QuantumCircuit(self._variable_register)

    def interpret_measurement(self, measurement, *args, **kwargs):
        top_measurement = max(
            measurement.items(), key=operator.itemgetter(1))[0]
        top_measurement = int(top_measurement)
        if top_measurement == 0:
            return "constant"
        else:
            return "balanced"

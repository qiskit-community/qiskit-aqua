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

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import ESOP
from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


class ESOPOracle(Oracle):

    CONFIGURATION = {
        'name': 'ESOPOracle',
        'description': 'Exclusive Sum of Products Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'esop_oracle_schema',
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

        # check that all outputs are of the same length
        out_len = len(list(bitmap.values())[0])
        for val in bitmap.values():
            if not len(val) == out_len:
                raise AquaError('The bitmap output lengths are not consistent.')

        # TODO: somehow move this following checks to DJ
        # # checks the input bitstring represents a constant or balanced function
        # bitsum = sum([int(bit) for bit in bitmap.values()])
        #
        # if bitsum == 0 or bitsum == 2 ** nbits:
        #     pass  # constant
        # elif bitsum == 2 ** (nbits - 1):
        #     pass  # balanced
        # else:
        #     raise AquaError('Input is not a balanced or constant function.')

        def _(bbs):
            return [i[-1] if i[0] == '1' else -i[-1] for i in list(zip(bbs, list(range(1, len(bbs) + 1))))]

        esop_exprs = []
        for out_idx in range(out_len):
            esop_expr = [_(i) for i in bitmap if bitmap[i][out_idx] == '1']
            if esop_expr:
                esop_exprs.append(esop_expr)

        if esop_exprs:
            self._esops = [ESOP(esop_expr) for esop_expr in esop_exprs]
            self._outcome_register = QuantumRegister(out_len, name='o')
            self._circuit = self._esops[0].construct_circuit(qr_outcome=self._outcome_register)
            self._variable_register = self._esops[0].qr_variable
            self._ancillary_register = self._esops[0].qr_ancilla
        else:
            self._esops = None
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
        if self._esops:
            for idx in range(1, len(self._esops)):
                esop_circuit = self._esops[idx].construct_circuit(
                    qr_variable=self._variable_register,
                    qr_ancilla=self._ancillary_register,
                    qr_outcome=self._outcome_register,
                    outcome_idx=idx,
                    mct_mode=self._mct_mode
                )
                self._circuit += esop_circuit
            return self._circuit
        else:
            return QuantumCircuit(self._variable_register)

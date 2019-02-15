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
"""
The Truth Table-based Quantum Oracle.
"""

import logging
import math
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import ESOP
from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


def is_power_of_2(num):
    return num != 0 and ((num & (num - 1)) == 0)


class TruthTableOracle(Oracle):

    CONFIGURATION = {
        'name': 'TruthTableOracle',
        'description': 'Truth Table Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'truth_table_oracle_schema',
            'type': 'object',
            'properties': {
                'bitmaps': {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        {
                            "type": "string"
                        }
                    ]
                }
,
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

    def __init__(self, bitmaps, mct_mode='basic'):
        """
        Constructor for Truth Table-based Oracle

        Args:
            bitmaps (str or [str]): A single binary string or a list of binary strings representing the desired
                single- and multi-value truth table.
            mct_mode (str): The mode to use when constructing multiple-control Toffoli
        """
        self.validate(locals())
        self._mct_mode = mct_mode

        if isinstance(bitmaps, str):
            bitmaps = [bitmaps]
        else:
            if not isinstance(bitmaps, list):
                raise AquaError('Bitmaps must be a single bitstring or a list of bitstrings.')

        # check that the input bitmaps length is a power of 2
        if not is_power_of_2(len(bitmaps[0])):
            raise AquaError('Length of any bitmap must be a power of 2.')
        for bitmap in bitmaps[1:]:
            if not len(bitmap) == len(bitmaps[0]):
                raise AquaError('Length of all bitmaps must be the same.')
        nbits = int(math.log(len(bitmaps[0]), 2))

        out_len = len(bitmaps)

        esop_exprs = []
        for bitmap in bitmaps:
            # ones = [i for i, x in enumerate(bitmap) if x == '1']
            esop_expr = [
                [
                    x[1] * (-1 if x[0] == '0' else 1) for x in zip(e, reversed(range(1, nbits + 1)))
                ]
                for e in [
                    np.binary_repr(idx, nbits) for idx, v in enumerate(bitmap) if v == '1'
                ]
            ]
            if esop_expr:
                esop_exprs.append(esop_expr)

        if esop_exprs:
            self._esops = [ESOP(esop_expr) for esop_expr in esop_exprs]
            self._output_register = QuantumRegister(out_len, name='o')
            self._circuits = [self._esops[0].construct_circuit(output_register=self._output_register)]
            self._variable_register = self._esops[0].variable_register
            self._ancillary_register = self._esops[0].ancillary_register
        else:
            self._esops = None
            self._variable_register = QuantumRegister(nbits, name='v')
            self._output_register = QuantumRegister(1, name='o')
            self._ancillary_register = None

        super().__init__()

    @property
    def variable_register(self):
        return self._variable_register

    @property
    def ancillary_register(self):
        return self._ancillary_register

    @property
    def output_register(self):
        return self._output_register

    def construct_circuit(self):
        circuit = QuantumCircuit()
        if self._esops:
            for idx in range(1, len(self._esops)):
                self._circuits.append(self._esops[idx].construct_circuit(
                    variable_register=self._variable_register,
                    ancillary_register=self._ancillary_register,
                    output_register=self._output_register,
                    output_idx=idx,
                    mct_mode=self._mct_mode
                ))
            for c in self._circuits:
                circuit += c
        else:
            circuit.add_register(self._variable_register)
        return circuit

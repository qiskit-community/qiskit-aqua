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
The DIMACS-based Quantum Oracle.
"""

import logging

from pyeda.boolalg.expr import ast2expr
from pyeda.parsing.dimacs import parse_cnf

from .logic_expr_oracle import LogicExpressionOracle
from .oracle import Oracle

logger = logging.getLogger(__name__)


class DimacsOracle(LogicExpressionOracle):

    CONFIGURATION = {
        'name': 'DimacsOracle',
        'description': 'Dimacs Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dimacs_oracle_schema',
            'type': 'object',
            'properties': {
                'dimacs_str': {
                    'type': 'string',
                },
                "optimization_mode": {
                    "anyOf": [
                        {
                            "type": "null",
                        },
                        {
                            "type": "string",
                            "default": "espresso",
                            'oneOf': [
                                {
                                    'enum': [
                                        'espresso',
                                    ]
                                }
                            ]
                        }
                    ]
                },
                'mct_mode': {
                    'type': 'string',
                    'default': 'basic',
                    'oneOf': [
                        {
                            'enum': [
                                'basic',
                                'advanced',
                                'noancilla'
                            ]
                        }
                    ]
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, dimacs_str=None, optimization_mode=None, mct_mode='basic'):
        """
        Constructor.

        Args:
            dimacs_str (str): The string representation in DIMACS format.
            optimization_mode (string): Optimization mode to use for minimizing the circuit.
                Currently, besides no optimization if omitted, Aqua also supports an 'expresso' mode
                <https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer>
            mct_mode (str): The mode to use for building Multiple-Control Toffoli.
        """

        self.validate(locals())
        self._mct_mode = mct_mode
        self._optimization_mode = optimization_mode

        if dimacs_str is None:
            raise ValueError('Missing input DIMACS string.')

        expr = ast2expr(parse_cnf(dimacs_str.strip(), varname='v'))
        self._expr = expr
        self._process_expr()

        Oracle.__init__(self)

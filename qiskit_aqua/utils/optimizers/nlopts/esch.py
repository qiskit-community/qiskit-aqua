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

from qiskit_aqua.utils.optimizers import Optimizer
from ._nloptimizer import minimize
import logging

try:
    import nlopt
except ImportError:
    raise ImportWarning('nlopt cannot be imported')


logger = logging.getLogger(__name__)


class ESCH(Optimizer):
    """ESCH (evolutionary algorithm)

    NLopt global optimizer, derivative-free
    http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#esch-evolutionary-algorithm
    """

    ESCH_CONFIGURATION = {
        'name': 'ESCH',
        'description': 'GN_ESCH Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'esch_schema',
            'type': 'object',
            'properties': {
                'max_evals': {
                    'type': 'integer',
                    'default': 1000
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['max_evals'],
        'optimizer': ['global']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.ESCH_CONFIGURATION.copy())

    def init_args(self):
        pass

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        return minimize(nlopt.GN_ESCH, objective_function, variable_bounds, initial_point, **self._options)

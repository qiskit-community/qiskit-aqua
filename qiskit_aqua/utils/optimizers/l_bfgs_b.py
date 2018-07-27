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

from scipy import optimize as sciopt

from qiskit_aqua.utils.optimizers import Optimizer

logger = logging.getLogger(__name__)


class L_BFGS_B(Optimizer):
    """Limited-memory BFGS algorithm.

    Uses scipy.optimize.fmin_l_bfgs_b
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    """

    L_BFGS_B_CONFIGURATION = {
        'name': 'L_BFGS_B',
        'description': 'L_BFGS_B Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'l_bfgs_b_schema',
            'type': 'object',
            'properties': {
                'maxfun': {
                    'type': 'integer',
                    'default': 1000
                },
                'factr': {
                    'type': 'integer',
                    'default': 10
                },
                'iprint': {
                    'type': 'integer',
                    'default': -1
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxfun', 'factr', 'iprint'],
        'optimizer': ['local']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.L_BFGS_B_CONFIGURATION.copy())

    def init_args(self):
        pass

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        approx_grad = True if gradient_function is None else False
        sol, opt, info = sciopt.fmin_l_bfgs_b(objective_function, initial_point, bounds=variable_bounds,
                                              fprime=gradient_function, approx_grad=approx_grad, **self._options)
        return sol, opt, info['funcalls']

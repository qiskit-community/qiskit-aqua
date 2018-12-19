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

from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer

from qiskit_aqua.algorithms.components.optimizers import Optimizer
import numpy as np
import copy
logger = logging.getLogger(__name__)


class SGD(Optimizer):
    """Gradient Descent Optimization

    Uses sklearn.neural_network._stochastic_optimizers
    See https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/_stochastic_optimizers.py
    """

    SGD_CONFIGURATION = {
        'name': 'SGD',
        'description': 'Stochastic Gradient Descent Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'cobyla_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 100
                },
                'ftol': {
                    'type': 'number',
                    'default': 1e-06
                },
                'learning_rate_init': {
                    'type': 'number',
                    'default': .1
                },
                'lr_schedule': {
                    'type': 'number',
                    'default': 1e-06
                },
                'momentum': {
                    'type': 'number',
                    'default': 0.5
                },
                'nesterov': {
                    'type': 'boolean',
                    'default': True
                },
                'power_t': {
                    'type': 'number',
                    'default': 0.5
                },
                'entropy': {
                    'type': 'number',
                    'default': 0.0
                },
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        # 'options': ['maxiter', 'ftol', 'lr_schedule', 'learning_rate_init', 'momentum', 'nesterov', 'power_t', 'entropy'],
        'optimizer': ['local']
    }

    #Same defaults as sklearn uses
    def __init__(self, configuration=None):
        super().__init__(configuration or self.SGD_CONFIGURATION.copy())
        self._maxiter = 100
        self._ftol = 1e-06
        self._learning_rate_init = 0.1
        self._lr_schedule = 'constant'
        self._momentum = 0.9
        self._nesterov = True
        self._power_t = 0.5
        self._entropy = 0

    def init_args(self, maxiter, ftol, lr_schedule, learning_rate_init, momentum, nesterov, power_t, entropy):
        self._maxiter = maxiter
        self._ftol = ftol
        self._lr_schedule = lr_schedule
        self._learning_rate_init = learning_rate_init
        self._momentum = momentum
        self._nesterov = nesterov
        self._power_t = power_t
        self._entropy = 0

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)
        point = np.array(initial_point)
        learning_rates = np.ones(len(point))*self._learning_rate_init
        cost = objective_function(point)
        iteration = 0
        walk_distance = 0
        logger.debug("Starting cost: {0:f}".format(cost))

        for iteration in range(self._maxiter):
            for dimension in range(len(point)):
                step_up = (1 + (np.random.random() * self._entropy)) * learning_rates[dimension]
                step_down = (1 + (np.random.random() * self._entropy)) * learning_rates[dimension]
                up_pt = [point[j] + step_up if j == dimension else point[j] for j in range(len(point))]
                down_pt = [point[j] - step_down if j == dimension else point[j] for j in range(len(point))]
                up_cost = objective_function(up_pt)
                down_cost = objective_function(down_pt)
                if up_cost < cost and up_cost < down_cost:
                    point = up_pt
                    cost = up_cost
                    distance = step_up
                elif down_cost < cost:
                    point = down_pt
                    cost = down_cost
                    distance = step_down
                else:
                    learning_rates[dimension] *= .8
                    distance = 0
                walk_distance += distance
                logger.debug("Iteration: {0}, dimension: {1}, cost: {2:f}, up_cost: {3:f}, down_cost: {4:f}, "
                             "distance traveled: {5}".
                             format(iteration, dimension, cost, up_cost, down_cost, distance))
            if np.average(learning_rates) < self._ftol: break
        logger.debug("Cartesian distance travelled: {}".format(np.linalg.norm(point - initial_point)))
        logger.debug("Total walk distance travelled: {}".format(walk_distance))
        logger.debug("Starting point: " + point.__str__())
        logger.debug("End point: " + initial_point.__str__())
        return point, cost, iteration
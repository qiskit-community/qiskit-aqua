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
import numpy as np
from qiskit.aqua.components.gradients import Gradient

logger = logging.getLogger(__name__)


class ObjectiveFuncGradient(Gradient):
    CONFIGURATION = {
        'name': 'ObjectiveFuncGradient',
        'description': 'ObjectiveFuncGradient',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'gradient_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, params=None):
        super().__init__()



    def get_gradient_function(self, objective_function):
        """Return the gradient function for computing the gradient at a point
        Args:
            objective_function (func): the objective/cost function
        Returns:
            func: the gradient function for computing the gradient at a point
        """
        self.objective_function = objective_function

        def gradient_num_diff(x_center):
            """
            We compute the gradient with the numeric differentiation  around the point x_center.
            Args:
                x_center (ndarray): point around which we compute the gradient
            Returns:
                grad: the gradient computed
            """
            epsilon = 1e-8
            forig = self.objective_function(*((x_center,)))
            grad = []
            ei = np.zeros((len(x_center),), float)
            todos = []
            for k in range(len(x_center)):
                ei[k] = 1.0
                d = epsilon * ei
                todos.append(x_center + d)
                deriv = (self.objective_function(x_center + d) - forig)/epsilon
                grad.append(deriv)
                ei[k] = 0.0
            return np.array(grad)

        return gradient_num_diff # func returned






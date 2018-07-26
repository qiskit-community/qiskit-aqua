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

from qiskit_aqua.utils.optimizers import Optimizer

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation algorithm."""
    SPSA_CONFIGURATION = {
        'name': 'SPSA',
        'description': 'SPSA Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'spsa_schema',
            'type': 'object',
            'properties': {
                'max_trials': {
                    'type': 'integer',
                    'default': 1000
                },
                'save_steps': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'last_avg': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'parameters': {
                    'type': ['array', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['save_steps', 'last_avg'],
        'optimizer': ['local', 'noise']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SPSA_CONFIGURATION.copy())
        self._max_trials = None
        self._parameters = None

    def init_args(self, max_trials=1000, parameters=None):
        self._max_trials = max_trials
        self._parameters = parameters

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):

        if not isinstance(initial_point, np.ndarray):
            initial_point = np.asarray(initial_point)

        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        initial_c = 0.1
        target_update = 2*np.pi*0.1
        if self._parameters is None: # at least one calibration, at most 25 calibrations
            logger.debug('SPSA parameters is manually set, skip calibration.')
            num_steps_calibration = min(25, max(1, self._max_trials // 5))
            self._calibration(objective_function, initial_point, initial_c, target_update, num_steps_calibration)
        opt, sol, cplus, cminus, tplus, tminus = self._optimization(objective_function, initial_point,
                                                                    max_trials=self._max_trials, **self._options)
        return sol, opt, None

    def _optimization(self, obj_fun, initial_theta, max_trials, save_steps=1, last_avg=1):
        """Minimizes obj_fun(theta) with a simultaneous perturbation stochastic
        approximation algorithm.

        Args:
            obj_fun (callable): the function to minimize
            initial_theta (numpy.array): initial value for the variables of
                obj_fun
            max_trials (int) : the maximum number of trial steps ( = function
                calls/2) in the optimization
            save_steps (int) : stores optimization outcomes each 'save_steps'
                trial steps
            last_avg (int) : number of last updates of the variables to average
                on for the final obj_fun
        Returns:
            list: a list with the following elements:
                cost_final : final optimized value for obj_fun
                theta_best : final values of the variables corresponding to
                    cost_final
                cost_plus_save : array of stored values for obj_fun along the
                    optimization in the + direction
                cost_minus_save : array of stored values for obj_fun along the
                    optimization in the - direction
                theta_plus_save : array of stored variables of obj_fun along the
                    optimization in the + direction
                theta_minus_save : array of stored variables of obj_fun along the
                    optimization in the - direction
        """

        theta_plus_save = []
        theta_minus_save = []
        cost_plus_save = []
        cost_minus_save = []
        theta = initial_theta
        theta_best = np.zeros(initial_theta.shape)
        for k in range(max_trials):
            # SPSA Parameters
            a_spsa = float(self._parameters[0]) / np.power(k + 1 + self._parameters[4], self._parameters[2])
            c_spsa = float(self._parameters[1]) / np.power(k + 1, self._parameters[3])
            delta = 2 * np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
            # plus and minus directions
            theta_plus = theta + c_spsa * delta
            theta_minus = theta - c_spsa * delta
            # cost function for the two directions
            cost_plus = obj_fun(theta_plus)
            cost_minus = obj_fun(theta_minus)
            # derivative estimate
            g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
            # updated theta
            theta = theta - a_spsa * g_spsa
            # saving
            if k % save_steps == 0:
                logger.debug('Objective function at theta+ for step # {}: {:.7f}'.format(k, cost_plus))
                logger.debug('Objective function at theta- for step # {}: {:.7f}'.format(k, cost_minus))
                theta_plus_save.append(theta_plus)
                theta_minus_save.append(theta_minus)
                cost_plus_save.append(cost_plus)
                cost_minus_save.append(cost_minus)
                # logger.debug('objective function at for step # {}: {:.7f}'.format(k, obj_fun(theta)))

            if k >= max_trials - last_avg:
                theta_best += theta / last_avg
        # final cost update
        cost_final = obj_fun(theta_best)
        logger.debug('Final objective function is: %.7f' % cost_final)

        return [cost_final, theta_best, cost_plus_save, cost_minus_save,
                theta_plus_save, theta_minus_save]

    def _calibration(self, obj_fun, initial_theta, initial_c, target_update, stat):
        """Calibrates and stores the SPSA parameters back.

        Args:
            obj_fun (callable): the function to minimize.
            initial_theta (numpy.array): initial value for the variables of
                obj_fun.
            initial_c (float) : first perturbation of initial_theta.
            target_update (float) : the aimed update of variables on the first
                trial step.
            stat (int) : number of random gradient directions to average on in
                the calibration.
        """

        SPSA_parameters = np.zeros((5))
        SPSA_parameters[1] = initial_c
        SPSA_parameters[2] = 0.602
        SPSA_parameters[3] = 0.101
        SPSA_parameters[4] = 0
        delta_obj = 0
        logger.debug("Calibration...")
        for i in range(stat):
            if i % 5 == 0:
                logger.debug('calibration step # {} of {}'.format(str(i), str(stat)))
            delta = 2 * np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
            obj_plus = obj_fun(initial_theta + initial_c * delta)
            obj_minus = obj_fun(initial_theta - initial_c * delta)
            delta_obj += np.absolute(obj_plus - obj_minus) / stat

        SPSA_parameters[0] = target_update * 2 / delta_obj \
            * SPSA_parameters[1] * (SPSA_parameters[4] + 1)

        logger.debug('Calibrated SPSA_parameters[0] is %.7f' % SPSA_parameters[0])
        self._parameters = SPSA_parameters

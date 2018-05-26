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

import numpy as np
import logging

logger = logging.getLogger(__name__)


def SPSA_parameters(): # pre-computed
    SPSA_params = np.zeros((5))
    SPSA_params[0] = 4
    SPSA_params[1] = 0.1
    SPSA_params[2] = 0.602
    SPSA_params[3] = 0.101
    SPSA_params[4] = 0
    return SPSA_params


def SPSA_calibration(obj_fun, initial_theta, initial_c, target_update, stat):
    """Calibrates the first SPSA parameter.
    The calibration is chosen such that the first theta update is on average
    (with statistics regulated by stat) equivalent to target_update, given
    an initial_c (=SPSA_parameters[1]) value.

    Returns all 5 SPSA_parameters:

    SPSA_parameters[0] -> calibrated
    SPSA_parameters[1] -> input by user (initial_c)
    SPSA_parameters[2] -> fixed at 0.602
    SPSA_parameters[3] -> fixed at 0.101
    SPSA_parameters[4] -> fixed at 0
    """

    SPSA_parameters = np.zeros((5))
    SPSA_parameters[1] = initial_c
    SPSA_parameters[2] = 0.602
    SPSA_parameters[3] = 0.101
    SPSA_parameters[4] = 0

    Delta_obj = 0
    for i in range(stat):

        if i % 5 == 0:
            logger.debug('calibration step # '+str(i)+' of '+str(stat))

        Delta = 2*np.random.randint(2, size=np.shape(initial_theta)[0]) - 1

        obj_plus = obj_fun(initial_theta+initial_c*Delta)[0]
        obj_minus = obj_fun(initial_theta-initial_c*Delta)[0]

        Delta_obj += np.absolute(obj_plus - obj_minus)/stat

    SPSA_parameters[0] = target_update*2/Delta_obj*SPSA_parameters[1]*(SPSA_parameters[4]+1)

    logger.debug('calibrated SPSA_parameters[0] is '+str(SPSA_parameters[0]))

    return SPSA_parameters


def SPSA_optimization(obj_fun, initial_theta, SPSA_parameters, max_trials, save_steps = 10):
    """Minimize the obj_fun(controls).

    initial_theta = the intial controls
    SPSA_parameters = the numerical parameters
    max_trials = the maximum number of trials
    """
    theta_plus_save = []
    theta_minus_save = []
    cost_plus_save = []
    cost_minus_save = []
    theta = initial_theta
    for k in range(max_trials):
        # SPSA Paramaters
        a_spsa = float(SPSA_parameters[0])/np.power(k+1+SPSA_parameters[4], SPSA_parameters[2])
        c_spsa = float(SPSA_parameters[1])/np.power(k+1, SPSA_parameters[3])
        Delta = 2*np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        # plus and minus directions
        theta_plus = theta + c_spsa*Delta
        theta_minus = theta - c_spsa*Delta
        # cost fuction for two directions
        cost_plus = obj_fun(theta=theta_plus)[0]
        cost_minus = obj_fun(theta=theta_minus)[0]
        # derivative estimate
        g_spsa = (cost_plus - cost_minus)*Delta/(2.0*c_spsa)
        # updated theta
        theta = theta - a_spsa*g_spsa
        # saving
        if k % save_steps == 0:
            logger.debug('Step # ' + str(k) + ' o+: ',cost_plus, 'o-:', cost_minus)
            theta_plus_save.append(theta_plus)
            theta_minus_save.append(theta_minus)
            cost_plus_save.append(cost_plus)
            cost_minus_save.append(cost_minus)
    # final cost update
    cost_final = obj_fun(theta)[0]
    logger.debug('Final Cost is: ' + str(cost_final))
    return cost_final, theta, cost_plus_save, cost_minus_save, theta_plus_save, theta_minus_save

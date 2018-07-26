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

import multiprocessing
import platform
import logging

import numpy as np
from scipy import optimize as sciopt

from qiskit_aqua.utils.optimizers import Optimizer

logger = logging.getLogger(__name__)


class P_BFGS(Optimizer):
    """Limited-memory BFGS algorithm. Parallel instantiations.

    Uses scipy.optimize.fmin_l_bfgs_b
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    """

    P_BFGS_CONFIGURATION = {
        'name': 'P_BFGS',
        'description': 'Parallelized l_bfgs_b Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'p_bfgs_b_schema',
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
                },
                'max_processes': {
                    'type': ['integer', 'null'],
                    'minimum': 1,
                    'default': None
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
        'optimizer': ['local', 'parallel']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.P_BFGS_CONFIGURATION.copy())
        self._max_processes = None

    def init_args(self, max_processes=None):
        self._max_processes = max_processes
        pass

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        num_procs = multiprocessing.cpu_count() - 1
        num_procs = num_procs if self._max_processes is None else min(num_procs, self._max_processes)
        num_procs = num_procs if num_procs >= 0 else 0

        if platform.system() == "Windows":
            num_procs = 0
            logger.warning("Using only current process. Multiple core use not supported in Windows")

        queue = multiprocessing.Queue()
        threshold = 2*np.pi  # bounds for additional initial points in case bounds has any None values
        low = [(l if l is not None else -threshold) for (l, u) in variable_bounds]
        high = [(u if u is not None else threshold) for (l, u) in variable_bounds]

        def optimize_runner(_queue, _i_pt):  # Multi-process sampling
            _sol, _opt, _nfev = self._optimize(num_vars, objective_function, gradient_function, variable_bounds, _i_pt)
            _queue.put((_sol, _opt, _nfev))

        # Start off as many other processes running the optimize (can be 0)
        processes = []
        for i in range(num_procs):
            i_pt = np.random.uniform(low, high)  # Another random point in bounds
            p = multiprocessing.Process(target=optimize_runner, args=(queue, i_pt))
            processes.append(p)
            p.start()

        # While the one _optimize in this process below runs the other processes will be running to. This one runs
        # with the supplied initial point. The process ones have their own random one
        sol, opt, nfev = self._optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        for p in processes:
            # For each other process we wait now for it to finish and see if it has a better result than above
            p.join()
            p_sol, p_opt, p_nfev = queue.get()
            if p_opt < opt:
                sol, opt = p_sol, p_opt
            nfev += p_nfev

        return sol, opt, nfev

    def _optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        approx_grad = True if gradient_function is None else False
        sol, opt, info = sciopt.fmin_l_bfgs_b(objective_function, initial_point, bounds=variable_bounds,
                                              fprime=gradient_function, approx_grad=approx_grad, **self._options)
        return sol, opt, info['funcalls']

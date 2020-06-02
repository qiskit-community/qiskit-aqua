# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

# from scipy.optimize import minimize
import skquant.opt as skq
from SQSnobFit import optset
from typing import Optional


from qiskit.aqua.components.optimizers import Optimizer
import numpy as np

logger = logging.getLogger(__name__)


class SNOBFIT(Optimizer):
    """Constrained Optimization By Linear Approximation algorithm.

    Uses scipy.optimize.minimize COBYLA
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'SNOBFIT',
        'description': 'SNOBFIT Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'cobyla_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 10000
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'rhobeg': {
                    'type': 'number',
                    'default': 1.0
                },
                'tol': {
                    'type': ['number', 'null'],
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
        'options': ['maxiter'],
        'optimizer': ['local']
    }

    _OPTIONS = ['maxiter', 'disp', 'rhobeg']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: int = 10000,
                 disp: bool = False,
                 rhobeg: float = 1.0,
                 tol: Optional[float] = None) -> None:
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter: Maximum number of function evaluations.
            disp: Set to True to print convergence messages.
            rhobeg: Reasonable initial changes to the variables.
            tol: Final accuracy in the optimization (not precisely guaranteed).
                         This is a lower bound on the size of the trust region.
        """
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        # self._tol = tol
        self._maxiter = maxiter

    def get_support_level(self):
        """ return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)
        options = optset()
        variable_bounds = []
        bound = 2*np.pi
        for _ in range(len(initial_point)):
            variable_bounds.append([-bound, bound])
        variable_bounds = np.array(variable_bounds, dtype=float)
        # counters the error when initial point is outside the acceptable bounds
        for idx, theta in enumerate(initial_point):
            if abs(theta) > bound:
                initial_point[idx] = initial_point[idx] % bound

        res, history = skq.minimize(objective_function, np.array(initial_point, dtype=float),
                                    bounds=variable_bounds, budget=100000,
                                    method="snobfit", options=options)
        return res.optpar, res.optval, len(history)

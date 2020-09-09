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
import skquant.opt as skq
from .optimizer import Optimizer, OptimizerSupportLevel

logger = logging.getLogger(__name__)


class IMFIL(Optimizer):
    """IMplicit FILtering algorithm.

    Implicit filtering is a way to solve bound-constrained optimization problems for
    which derivatives are not available. In comparison to methods that use interpolation to
    reconstruct the function and its higher derivatives, implicit filtering builds upon
    coordinate search followed by interpolation to get an approximate gradient.

    Uses scipy.optimize.minimize IMFIL
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: int = 1000,
                 ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.
        """
        super().__init__()
        self._maxiter = maxiter

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.required,
            'initial_point': OptimizerSupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None,
                 initial_point=None):
        """ Runs the optimization """
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds,
                         initial_point)
        res, history = skq.minimize(func=objective_function, x0=initial_point,
                                    bounds=variable_bounds, budget=self._maxiter,
                                    method="imfil")
        return res.optpar, res.optval, len(history)

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError


class QualityMetrics(object):
    """A class containing measures of the quality of a solution.

    The __str__ method of this class prints all available measures of
    the quality of the solution in human readable form.

    This class may have a different set of data members depending on
    the optimization algorithm used and the quality metrics that are
    available.

    An instance of this class always has the member quality_type,
    which is one of the following strings:

    "quadratically_constrained"
    "MIP"

    If self.quality_type is "quadratically_constrained" this instance
    has the following members:

    objective
    norm_total
    norm_max
    error_Ax_b_total
    error_Ax_b_max
    error_xQx_dx_f_total
    error_xQx_dx_f_max
    x_bound_error_total
    x_bound_error_max
    slack_bound_error_total
    slack_bound_error_max
    quadratic_slack_bound_error_total
    quadratic_slack_bound_error_max
    normalized_error_max

    If self.quality_type is "MIP" and this instance was generated for
    the incumbent solution, it has the members:

    solver
    objective
    x_norm_total
    x_norm_max
    error_Ax_b_total
    error_Ax_b_max
    x_bound_error_total
    x_bound_error_max
    integrality_error_total
    integrality_error_max
    slack_bound_error_total
    slack_bound_error_max

    If solver is "MIQCP" this instance also has the members:

    error_xQx_dx_f_total
    error_xQx_dx_f_max
    quadratic_slack_bound_error_total
    quadratic_slack_bound_error_max
    """

    def __init__(self, soln=-1):
        self._tostring = ""

    def __str__(self):
        # See __init__ (above) to see how this is constructed.
        return self._tostring

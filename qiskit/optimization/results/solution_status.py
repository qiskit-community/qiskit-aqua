# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Solution status codes."""

from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError

CPX_STAT_ABORT_DETTIME_LIM = 25
CPX_STAT_ABORT_IT_LIM = 10
CPX_STAT_ABORT_OBJ_LIM = 12
CPX_STAT_ABORT_TIME_LIM = 11
CPX_STAT_ABORT_USER = 13
CPX_STAT_FEASIBLE = 23
CPX_STAT_INFEASIBLE = 3
CPX_STAT_INF_OR_UNBD = 4
CPX_STAT_OPTIMAL = 1
CPX_STAT_UNBOUNDED = 2

CPXMIP_OPTIMAL = 101
CPXMIP_INFEASIBLE = 103
CPXMIP_TIME_LIM_FEAS = 107
CPXMIP_TIME_LIM_INFEAS = 108
CPXMIP_FAIL_FEAS = 109
CPXMIP_FAIL_INFEAS = 110
CPXMIP_MEM_LIM_FEAS = 111
CPXMIP_MEM_LIM_INFEAS = 112
CPXMIP_ABORT_FEAS = 113
CPXMIP_ABORT_INFEAS = 114
CPXMIP_UNBOUNDED = 118
CPXMIP_INF_OR_UNBD = 119
CPXMIP_FEASIBLE = 127
CPXMIP_DETTIME_LIM_FEAS = 131
CPXMIP_DETTIME_LIM_INFEAS = 132


class SolutionStatus:
    """Solution status codes.

    For documentation of each status code, see the reference manual
    of the CPLEX Callable Library, especially the group
    optim.cplex.callable.solutionstatus.
    """
    unknown = 0  # There is no constant for this.
    optimal = CPX_STAT_OPTIMAL
    unbounded = CPX_STAT_UNBOUNDED
    infeasible = CPX_STAT_INFEASIBLE
    feasible = CPX_STAT_FEASIBLE
    infeasible_or_unbounded = CPX_STAT_INF_OR_UNBD
    abort_obj_limit = CPX_STAT_ABORT_OBJ_LIM
    abort_iteration_limit = CPX_STAT_ABORT_IT_LIM
    abort_time_limit = CPX_STAT_ABORT_TIME_LIM
    abort_dettime_limit = CPX_STAT_ABORT_DETTIME_LIM
    abort_user = CPX_STAT_ABORT_USER
    fail_feasible = CPXMIP_FAIL_FEAS
    fail_infeasible = CPXMIP_FAIL_INFEAS
    mem_limit_feasible = CPXMIP_MEM_LIM_FEAS
    mem_limit_infeasible = CPXMIP_MEM_LIM_INFEAS

    MIP_optimal = CPXMIP_OPTIMAL
    MIP_infeasible = CPXMIP_INFEASIBLE
    MIP_time_limit_feasible = CPXMIP_TIME_LIM_FEAS
    MIP_time_limit_infeasible = CPXMIP_TIME_LIM_INFEAS
    MIP_dettime_limit_feasible = CPXMIP_DETTIME_LIM_FEAS
    MIP_dettime_limit_infeasible = CPXMIP_DETTIME_LIM_INFEAS
    MIP_abort_feasible = CPXMIP_ABORT_FEAS
    MIP_abort_infeasible = CPXMIP_ABORT_INFEAS
    MIP_unbounded = CPXMIP_UNBOUNDED
    MIP_infeasible_or_unbounded = CPXMIP_INF_OR_UNBD
    MIP_feasible = CPXMIP_FEASIBLE

    def __getitem__(self, item):
        """Converts a constant to a string.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.solution.status.optimal
        1
        >>> c.solution.status[1]
        'optimal'
        """
        solution_status = {
            0: 'unknown',
            CPX_STAT_OPTIMAL: 'optimal',
            CPX_STAT_UNBOUNDED: 'unbounded',
            CPX_STAT_INFEASIBLE: 'infeasible',
            CPX_STAT_FEASIBLE: 'feasible',
            CPX_STAT_INF_OR_UNBD: 'infeasible_or_unbounded',
            CPX_STAT_ABORT_OBJ_LIM: 'abort_obj_limit',
            CPX_STAT_ABORT_IT_LIM: 'abort_iteration_limit',
            CPX_STAT_ABORT_TIME_LIM: 'abort_time_limit',
            CPX_STAT_ABORT_DETTIME_LIM: 'abort_dettime_limit',
            CPX_STAT_ABORT_USER: 'abort_user',
            CPXMIP_FAIL_FEAS: 'fail_feasible',
            CPXMIP_FAIL_INFEAS: 'fail_infeasible',
            CPXMIP_MEM_LIM_FEAS: 'mem_limit_feasible',
            CPXMIP_MEM_LIM_INFEAS: 'mem_limit_infeasible',
            CPXMIP_OPTIMAL: 'MIP_optimal',
            CPXMIP_INFEASIBLE: 'MIP_infeasible',
            CPXMIP_TIME_LIM_FEAS: 'MIP_time_limit_feasible',
            CPXMIP_TIME_LIM_INFEAS: 'MIP_time_limit_infeasible',
            CPXMIP_DETTIME_LIM_FEAS: 'MIP_dettime_limit_feasible',
            CPXMIP_DETTIME_LIM_INFEAS: 'MIP_dettime_limit_infeasible',
            CPXMIP_ABORT_FEAS: 'MIP_abort_feasible',
            CPXMIP_ABORT_INFEAS: 'MIP_abort_infeasible',
            CPXMIP_UNBOUNDED: 'MIP_unbounded',
            CPXMIP_INF_OR_UNBD: 'MIP_infeasible_or_unbounded',
            CPXMIP_FEASIBLE: 'MIP_feasible'
        }
        try:
            return solution_status[item]
        except KeyError:
            raise QiskitOptimizationError('Unexpected solution status code!')

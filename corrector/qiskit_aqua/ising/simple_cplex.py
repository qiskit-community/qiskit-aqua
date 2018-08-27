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

"""Simple Python Wrapper for CPLEX"""

import logging
from itertools import product
from sys import stdout

from qiskit_aqua import AlgorithmError

try:
    from cplex import Cplex, SparsePair, SparseTriple
except ImportError:
    raise ImportWarning('CPLEX is not installed. See https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html')

logger = logging.getLogger(__name__)

class SimpleCPLEX:
    def __init__(self, cplex=None):
        if cplex:
            self._model = Cplex(cplex._model)
        else:
            self._model = Cplex()
        self._init_lin()
        # to avoid a variable with index 0
        self._model.variables.add(names=['_dummy_'], types=[self._model.variables.type.continuous])
        self._var_id = {'_dummy_': 0}

    def _init_lin(self):
        self._lin = {
            'lin_expr': [],
            'senses': [],
            'rhs': [],
            'range_values': [],
            'names': []
        }

    def register_variables(self, prefix, ranges, var_type, lb=None, ub=None):
        if not ranges:  # None or []
            return self._register_variable(prefix, var_type, lb, ub)

        variables = {}
        for keys in product(*ranges):
            name = '_'.join([prefix] + [str(e) for e in keys])
            index = self._register_variable(name, var_type, lb, ub)
            if len(keys) == 1:
                keys = keys[0]
            variables[keys] = index
        return variables

    def _register_variable(self, name, var_type, lb, ub):
        self._model.variables.add(names=[name], types=[var_type],
                                  lb=[] if lb is None else [lb],
                                  ub=[] if ub is None else [ub])
        if name in self._var_id:
            logger.info('Variable %s is already registered. Overwritten', name)
        index = len(self._var_id)
        self._var_id[name] = index
        return index

    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._model.parameters

    @property
    def variables(self):
        return self._model.variables

    @property
    def objective(self):
        return self._model.objective

    @property
    def problem_type(self):
        return self._model.problem_type

    @property
    def solution(self):
        return self._model.solution

    @property
    def version(self):
        return self._model.get_version()

    def maximize(self):
        self._model.objective.set_sense(self._model.objective.sense.maximize)

    def minimize(self):
        self._model.objective.set_sense(self._model.objective.sense.minimize)

    def set_problem_type(self, problem_type):
        self._model.set_problem_type(problem_type)

    def tune_problem(self, options):
        self._model.set_results_stream(None)
        self._model.parameters.tune_problem(options)
        self._model.set_results_stream(stdout)

    def solve(self):
        self._model.solve()

    def populate(self):
        self._model.solve()
        self._model.populate_solution_pool()

    def variable(self, name):
        """
        :param name: variable name
        :type name: str
        :return: variable index in CPLEX model
        :rtype: int
        """
        return self._var_id[name]

    def get_values(self, lst, idx=None):
        if idx:
            return self._model.solution.pool.get_values(idx, lst)
        else:
            return self._model.solution.get_values(lst)

    def get_objective_value(self, idx=None):
        if idx:
            return self._model.solution.pool.get_objective_value(idx)
        else:
            return self._model.solution.get_objective_value()

    @property
    def num_solutions(self):
        return self._model.solution.pool.get_num()

    @staticmethod
    def _convert_sense(sense):
        # Note: ignore 'R' range case
        assert sense in ['E', 'L', 'G', '>=', '=', '==', '<=']
        if sense == '<=':
            sense = 'L'
        elif sense == '=' or sense == '==':
            sense = 'E'
        elif sense == '>=':
            sense = 'G'
        return sense

    def set_objective(self, lst):
        """
        :type lst: list[int or (int, float) or (int, int, float)] or float
        """
        if isinstance(lst, float):
            self._model.objective.set_offset(lst)
            return

        linear = []
        quad = []
        assert isinstance(lst, list)
        for e in lst:
            assert isinstance(e, int) or isinstance(e, tuple)
            if isinstance(e, int):
                if e > 0:
                    linear.append((e, 1))
                elif e < 0:
                    linear.append((-e, -1))
                else:
                    raise RuntimeError('invalid variable ID')
            elif len(e) == 2:
                linear.append(e)
            else:
                assert len(e) == 3
                e = (min(e[0], e[1]), max(e[0], e[1]), e[2])
                quad.append(e)
        if linear:
            self._model.objective.set_linear(linear)
        if quad:
            self._model.objective.set_quadratic_coefficients(quad)

    @staticmethod
    def _convert_coefficients(coef):
        """
        Convert 'x', and '-x' into ('x', 1) and ('x', -1), respectively.

        :type coef: list[(int, float) or int]
        :rtype: (list[int], list[float])
        """
        ind = []
        val = []
        for e in coef:
            if isinstance(e, tuple):
                assert len(e) == 2
                ind.append(e[0])
                val.append(e[1])
            elif isinstance(e, int):
                if e >= 0:
                    ind.append(e)
                    val.append(1)
                else:
                    ind.append(-e)
                    val.append(-1)
            else:
                raise RuntimeError('unsupported type:' + str(e))
        return ind, val

    def add_linear_constraint(self, coef, sense, rhs):
        """
        :type coef: list[(int, float)]
        :type sense: string
        :type rhs: float
        :rtype: None
        """
        if not coef:
            logger.warning('empty linear constraint')
            return
        ind, val = self._convert_coefficients(coef)
        sense = self._convert_sense(sense)
        c = self._lin
        c['lin_expr'].append(SparsePair(ind, val))
        c['senses'].append(sense)
        c['rhs'].append(rhs)
        c['range_values'].append(0)
        c['names'].append('c' + str(len(self._lin['names'])))
        # logger.debug('%s %s %s %s', c['names'][-1], c['lin_expr'][-1], c['senses'][-1], c['rhs'][-1])

    def add_indicator_constraint(self, indvar, complemented, coef, sense, rhs):
        """
        :type indvar: int
        :type complemented: int
        :type coef: list[(int, float)]
        :type sense: string
        :type rhs: float
        :rtype: None
        """
        ind, val = self._convert_coefficients(coef)
        sense = self._convert_sense(sense)
        c = {'lin_expr': SparsePair(ind, val),
             'sense': sense,
             'rhs': rhs,
             'name': 'i' + str(self._model.indicator_constraints.get_num()),
             'indvar': indvar,
             'complemented': complemented
             }

        self._model.indicator_constraints.add(**c)

    def add_sos(self, coef):
        """
        :type coef: list[(int, float)]
        """
        ind, val = self._convert_coefficients(coef)
        c = {'type': '1',
             'SOS': SparsePair(ind, val),
             'name': 'sos' + str(self._model.SOS.get_num()),
             }

        self._model.SOS.add(**c)

    def add_quadratic_constraint(self, lin, quad, sense, rhs):
        """
        :type lin: list[(int, float)]
        :type quad: list[(int, int, float)]
        :type sense: string
        :type rhs: float
        :rtype: None
        """
        ind, val = self._convert_coefficients(lin)
        ind1 = [e[0] for e in quad]
        ind2 = [e[1] for e in quad]
        val2 = [e[2] for e in quad]
        sense = self._convert_sense(sense)
        c = {'lin_expr': SparsePair(ind, val),
             'quad_expr': SparseTriple(ind1, ind2, val2),
             'sense': sense,
             'rhs': rhs,
             'name': 'q' + str(self._model.quadratic_constraints.get_num())
             }

        self._model.quadratic_constraints.add(**c)

    def build_model(self):
        self._model.linear_constraints.add(**self._lin)
        self._init_lin()

    def write(self, filename, filetype=''):
        self._model.write(filename, filetype)

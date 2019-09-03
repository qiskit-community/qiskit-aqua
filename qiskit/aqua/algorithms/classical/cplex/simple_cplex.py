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

"""Simple Python Wrapper for CPLEX"""

import logging
from itertools import product
from sys import stdout

logger = logging.getLogger(__name__)

try:
    from cplex import Cplex, SparsePair, SparseTriple
except ImportError:
    logger.info('CPLEX is not installed. See https://www.ibm.com/support/knowledgecenter/'
                'SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html')

# pylint: disable=invalid-name


class SimpleCPLEX:
    """Simple Python Wrapper for CPLEX"""
    def __init__(self, cplex=None):
        try:
            if cplex:
                self._model = Cplex(cplex._model)
            else:
                self._model = Cplex()
        except NameError:
            raise NameError('CPLEX is not installed. '
                            'See https://www.ibm.com/support/knowledgecenter/'
                            'SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/'
                            'topics/COS_home.html')

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
        """ register variables """
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
        """ returns model """
        return self._model

    @property
    def parameters(self):
        """ returns parameters """
        return self._model.parameters

    @property
    def variables(self):
        """ returns variables """
        return self._model.variables

    @property
    def objective(self):
        """ returns objective """
        return self._model.objective

    @property
    def problem_type(self):
        """ returns problem type """
        return self._model.problem_type

    @property
    def solution(self):
        """ returns solution """
        return self._model.solution

    @property
    def version(self):
        """ returns version """
        return self._model.get_version()

    def maximize(self):
        """ maximize """
        self._model.objective.set_sense(self._model.objective.sense.maximize)

    def minimize(self):
        """ minimize """
        self._model.objective.set_sense(self._model.objective.sense.minimize)

    def set_problem_type(self, problem_type):
        """ set problem type """
        self._model.set_problem_type(problem_type)

    def tune_problem(self, options):
        """ tune problem """
        self._model.set_results_stream(None)
        self._model.parameters.tune_problem(options)
        self._model.set_results_stream(stdout)

    def solve(self):
        """ solve """
        self._model.solve()

    def populate(self):
        """ populate """
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
        """ get values """
        if idx:
            return self._model.solution.pool.get_values(idx, lst)
        else:
            return self._model.solution.get_values(lst)

    def get_objective_value(self, idx=None):
        """ get objective value """
        if idx:
            return self._model.solution.pool.get_objective_value(idx)
        else:
            return self._model.solution.get_objective_value()

    @property
    def num_solutions(self):
        """ returns num solutions """
        return self._model.solution.pool.get_num()

    @staticmethod
    def _convert_sense(sense):
        # Note: ignore 'R' range case
        assert sense in ['E', 'L', 'G', '>=', '=', '==', '<=']
        if sense == '<=':
            sense = 'L'
        elif sense in ('=', '=='):
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
            assert isinstance(e, (int, tuple))
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

        Args:
            coef (list[(int, float) or int]): coef
        Returns:
            tuple: ind, val
        Raises:
            RuntimeError: unsupported type
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
        Args:
            coef (list[(int, float)]): coef
            sense (str): sense
            rhs (float): rhs
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

    def add_indicator_constraint(self, indvar, complemented, coef, sense, rhs):
        """
        Args:
            indvar (int): ind var
            complemented (int): complemented
            coef (list[(int, float)]): coef
            sense (str): sense
            rhs (float): rhs
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
        Args:
            lin (list[(int, float)]): lin
            quad (list[(int, int, float)]): quad
            sense (str): sense
            rhs (float): rhs
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
        """ build model """
        self._model.linear_constraints.add(**self._lin)
        self._init_lin()

    def write(self, filename, filetype=''):
        """ write """
        self._model.write(filename, filetype)

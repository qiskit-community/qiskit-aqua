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

"""CPLEX Ising algorithm; uses IBM CPLEX backend for Ising Hamiltonian solution"""

import csv
import logging
from math import fsum
from timeit import default_timer
from typing import Dict, List, Tuple, Any
import importlib
import numpy as np

from qiskit.aqua import QuantumAlgorithm, Pluggable, AquaError
from qiskit.aqua.algorithms.classical.cplex.simple_cplex import SimpleCPLEX

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class CPLEX_Ising(QuantumAlgorithm):
    """ CPLEX Ising algorithm """
    CONFIGURATION = {
        'name': 'CPLEX.Ising',
        'description': 'CPLEX backend for Ising Hamiltonian',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'CPLEX_schema',
            'type': 'object',
            'properties': {
                'timelimit': {
                    'type': 'integer',
                    'default': 600,
                    'minimum': 1
                },
                'thread': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
                'display': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 0,
                    'maximum': 5
                }
            },
            'additionalProperties': False
        },
        'problems': ['ising']
    }

    def __init__(self, operator, timelimit=600, thread=1, display=2):
        self.validate(locals())
        super().__init__()
        self._ins = IsingInstance()
        self._ins.parse(operator.to_dict()['paulis'])
        self._timelimit = timelimit
        self._thread = thread
        self._display = display
        self._sol = None

    @classmethod
    def init_params(cls, params, algo_input):
        """ init params """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")
        algo_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        timelimit = algo_params['timelimit']
        thread = algo_params['thread']
        display = algo_params['display']
        return cls(algo_input.qubit_op, timelimit, thread, display)

    @staticmethod
    def check_pluggable_valid():
        """ check pluggable valid """
        err_msg = 'CPLEX is not installed. See ' \
            'https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/' \
            'ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html'
        try:
            spec = importlib.util.find_spec('cplex.callbacks')
            if spec is not None:
                spec = importlib.util.find_spec('cplex.exceptions')
                if spec is not None:
                    return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug('%s %s', err_msg, str(ex))
            raise AquaError(err_msg) from ex

        raise AquaError(err_msg)

    def _run(self):
        model = IsingModel(self._ins, timelimit=self._timelimit,
                           thread=self._thread, display=self._display)
        self._sol = model.solve()
        return {'energy': self._sol.objective, 'eval_time': self._sol.time,
                'x_sol': self._sol.x_sol, 'z_sol': self._sol.z_sol,
                'eigvecs': self._sol.eigvecs}

    @property
    def solution(self):
        """ return solution """
        return self._sol


def new_cplex(timelimit=600, thread=1, display=2):
    """ new cplex """
    cplex = SimpleCPLEX()
    cplex.parameters.timelimit.set(timelimit)
    cplex.parameters.threads.set(thread)
    cplex.parameters.mip.display.set(display)
    cplex.parameters.mip.tolerances.integrality.set(0)
    cplex.parameters.mip.tolerances.mipgap.set(0)
    return cplex


class IsingInstance:
    """ Ising Instance """
    def __init__(self):
        self._num_vars = 0
        self._const = 0
        self._lin = {}
        self._quad = {}

    @property
    def num_vars(self) -> int:
        """ returns number of vars """
        return self._num_vars

    @property
    def constant(self) -> float:
        """ returns constant """
        return self._const

    @property
    def linear_coef(self) -> Dict[int, float]:
        """ returns linear coefficient """
        return self._lin

    @property
    def quad_coef(self) -> Dict[Tuple[int, int], float]:
        """ returns quad coef """
        return self._quad

    def parse(self, pauli_list: List[Dict[str, Any]]):
        """ parse """
        for pauli in pauli_list:
            if self._num_vars == 0:
                self._num_vars = len(pauli['label'])
            elif self._num_vars != len(pauli['label']):
                logger.critical('Inconsistent number of qubits: (target) %d, (actual) %d %s',
                                self._num_vars, len(pauli['label']), pauli)
                continue
            label = pauli['label'][::-1]
            if 'imag' in pauli['coeff'] and pauli['coeff']['imag'] != 0.0:
                logger.critical(
                    'CPLEX backend cannot deal with complex coefficient %s', pauli)
                continue
            weight = pauli['coeff']['real']
            if 'X' in label or 'Y' in label:
                logger.critical(
                    'CPLEX backend cannot deal with X and Y Pauli matrices: %s', pauli)
                continue
            ones = []
            for i, e in enumerate(label):
                if e == 'Z':
                    ones.append(i)
            ones = np.array(ones)
            size = len(ones)
            if size == 0:
                if not isinstance(self._const, int):
                    logger.warning(
                        'Overwrite the constant: (current) %f, (new) %f', self._const, weight)
                self._const = weight
            elif size == 1:
                k = ones[0]
                if k in self._lin:
                    logger.warning('Overwrite the linear coefficient %s: (current) %f, (new) %f',
                                   k, self._lin[k], weight)
                self._lin[k] = weight
            elif size == 2:
                k = tuple(sorted(ones))
                if k in self._lin:
                    logger.warning('Overwrite the quadratic coefficient %s: (current) %f, (new) %f',
                                   k, self._lin[k], weight)
                self._quad[k] = weight
            else:
                logger.critical(
                    'CPLEX backend cannot deal with Hamiltonian more than quadratic: %s', pauli)


class IsingModel:
    """ Ising Model """
    def __init__(self, instance: IsingInstance, **kwargs):
        self._instance = instance
        self._num_vars = instance.num_vars
        self._cplex = new_cplex(**kwargs)
        self._const = instance.constant
        self._lin = instance.linear_coef
        self._quad = instance.quad_coef
        self._var_ids = range(instance.num_vars)

    @property
    def linear_coef(self):
        """ returns linear coef """
        return self._lin

    @property
    def quad_coef(self):
        """ returns quad coef """
        return self._quad

    def _register_variables(self):
        binary = self._cplex.variables.type.binary
        x = self._cplex.register_variables('x', [self._var_ids], binary)
        return x

    def _cost_objective(self, x):
        lin = {i: -2 * w for i, w in self._lin.items()}
        for (i, j), w in self._quad.items():
            if i not in lin:
                lin[i] = 0
            if j not in lin:
                lin[j] = 0
            lin[i] += -2 * w
            lin[j] += -2 * w
        self._cplex.set_objective([(x[i], float(w)) for i, w in lin.items()])
        self._cplex.set_objective([(x[i], x[j], float(4 * w))
                                   for (i, j), w in self._quad.items()])
        self._cplex.set_objective(
            fsum([self._const] + list(self._lin.values()) + list(self._quad.values())))

    def solve(self):
        """ solve """
        start = default_timer()
        x = self._register_variables()
        self._cplex.minimize()
        self._cost_objective(x)
        self._cplex.build_model()
        self._cplex.solve()
        # self._display()
        return self._solution(x, default_timer() - start)

    def _display(self):
        logger.debug('objective %f', self._cplex.get_objective_value())
        logger.debug(self._cplex.solution.get_quality_metrics())
        print('objective {}'.format(self._cplex.get_objective_value()))

    def _solution(self, x, elapsed):
        xx = self._cplex.get_values([x[i] for i in self._var_ids])
        sol = {b: int(xx[i]) for i, b in enumerate(self._var_ids)}
        return IsingSolution(self._instance, sol, elapsed, self._cplex.get_objective_value())


class IsingSolution:
    """ Ising Solution """
    delimiter = '\t'

    def __init__(self, ins: IsingInstance, sol: Dict[int, int], elapsed: float, obj):
        self._ins = ins
        self._x_sol = sol
        self._z_sol = {k: 1 - 2 * x for k, x in sol.items()}
        self._elapsed = elapsed
        self._obj = obj
        self._eigvecs = self._calc_eigvecs(sol)

    def feasible(self):
        """ solutions are always feasible because the problem is unconstrained """
        return True

    @staticmethod
    def _calc_eigvecs(sol):
        val = 0
        for k, v in sol.items():
            val += v * (2 ** k)
        ret = [0] * 2 ** len(sol)
        ret[val] = 1
        return np.array([ret])

    @property
    def eigvecs(self):
        """ returns eigvecs """
        return self._eigvecs

    @property
    def x_sol(self):
        """ returns x sol """
        return self._x_sol

    @property
    def z_sol(self):
        """ returns z sol """
        return self._z_sol

    def dump(self, filename):
        """ dump """
        with open(filename, 'w') as outfile:
            outfile.write('# objective {}\n'.format(self.objective))
            outfile.write('# elapsed time {}\n'.format(self._elapsed))
            writer = csv.writer(outfile, delimiter=self.delimiter)
            writer.writerow(['x_id', 'x_val'])
            for k, v in sorted(self._x_sol.items()):
                writer.writerow([k, v])

    @property
    def objective(self):
        """ returns objective """
        return self._obj

    @property
    def time(self):
        """ returns time elapsed """
        return self._elapsed

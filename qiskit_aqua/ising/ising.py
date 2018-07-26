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

"""CPLEX algorithm; uses IBM CPLEX backend for Ising Hamiltonian solution"""

import csv
import logging
from math import fsum
from timeit import default_timer
from typing import Dict, List, Tuple, Any
import copy
import numpy as np

from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua.ising.simple_cplex import SimpleCPLEX

logger = logging.getLogger(__name__)


class CPLEX(QuantumAlgorithm):
    CPLEX_CONFIGURATION = {
        'name': 'CPLEX',
        'description': 'CPLEX backend for Ising Hamiltonian',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
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

    def __init__(self, configuration=None):
        super().__init__(configuration or copy.deepcopy(CPLEX.CPLEX_CONFIGURATION))
        self._ins = IsingInstance()
        self._sol = None
        self._params = None

    def init_params(self, params, algo_input):
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")
        self._ins.parse(algo_input.qubit_op.save_to_dict()['paulis'])
        self._params = params

    def run(self):
        if self._params:
            timelimit = self._params['algorithm']['timelimit']
            thread = self._params['algorithm']['thread']
            display = self._params['algorithm']['display']
        else:
            timelimit = 600
            thread = 1
            display = 2
        model = IsingModel(self._ins, timelimit=timelimit, thread=thread, display=display)
        self._sol = model.solve()
        return {'energy': self._sol.objective, 'eval_time': self._sol.time,
                'x_sol': self._sol.x_sol, 'z_sol': self._sol.z_sol,
                'eigvecs': self._sol.eigvecs}

    @property
    def solution(self):
        return self._sol


def new_cplex(timelimit=600, thread=1, display=2):
    cplex = SimpleCPLEX()
    cplex.parameters.timelimit.set(timelimit)
    cplex.parameters.threads.set(thread)
    cplex.parameters.mip.display.set(display)
    cplex.parameters.mip.tolerances.integrality.set(0)
    cplex.parameters.mip.tolerances.mipgap.set(0)
    return cplex


class IsingInstance:
    def __init__(self):
        self._num_vars = 0
        self._const = 0
        self._lin = {}
        self._quad = {}

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def constant(self) -> float:
        return self._const

    @property
    def linear_coef(self) -> Dict[int, float]:
        return self._lin

    @property
    def quad_coef(self) -> Dict[Tuple[int, int], float]:
        return self._quad

    def parse(self, pauli_list: List[Dict[str, Any]]):
        for pauli in pauli_list:
            if self._num_vars == 0:
                self._num_vars = len(pauli['label'])
            elif self._num_vars != len(pauli['label']):
                logger.critical('Inconsistent number of qubits: (target) %d, (actual) %d %s', self._num_vars,
                                len(pauli['label']), pauli)
                continue
            label = pauli['label']
            if 'imag' in pauli['coeff'] and pauli['coeff']['imag'] != 0.0:
                logger.critical('CPLEX backend cannot deal with complex coefficient %s', pauli)
                continue
            weight = pauli['coeff']['real']
            if 'X' in label or 'Y' in label:
                logger.critical('CPLEX backend cannot deal with X and Y Pauli matrices: %s', pauli)
                continue
            ones = []
            for i, e in enumerate(label):
                if e == 'Z':
                    ones.append(i)
            ones = np.array(ones)
            size = len(ones)
            if size == 0:
                if not isinstance(self._const, int):
                    logger.warning('Overwrite the constant: (current) %f, (new) %f', self._const, weight)
                self._const = weight
            elif size == 1:
                k = ones[0]
                if k in self._lin:
                    logger.warning('Overwrite the linear coefficient %s: (current) %f, (new) %f', k, self._lin[k],
                                   weight)
                self._lin[k] = weight
            elif size == 2:
                k = tuple(sorted(ones))
                if k in self._lin:
                    logger.warning('Overwrite the quadratic coefficient %s: (current) %f, (new) %f', k, self._lin[k],
                                   weight)
                self._quad[k] = weight
            else:
                logger.critical('CPLEX backend cannot deal with Hamiltonian more than quadratic: %s', pauli)


class IsingModel:
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
        return self._lin

    @property
    def quad_coef(self):
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
        self._cplex.set_objective([(x[i], x[j], float(4 * w)) for (i, j), w in self._quad.items()])
        self._cplex.set_objective(fsum([self._const] + list(self._lin.values()) + list(self._quad.values())))

    def solve(self):
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
    delimiter = '\t'

    def __init__(self, ins: IsingInstance, sol: Dict[int, int], elapsed: float, obj):
        self._ins = ins
        self._x_sol = sol
        self._z_sol = {k: 1 - 2 * x for k, x in sol.items()}
        self._elapsed = elapsed
        self._obj = obj
        self._eigvecs = self._calc_eigvecs(sol)

    def feasible(self):
        # solutions are always feasible because the problem is unconstrained
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
        return self._eigvecs

    @property
    def x_sol(self):
        return self._x_sol

    @property
    def z_sol(self):
        return self._z_sol

    def dump(self, filename):
        with open(filename, 'w') as outfile:
            outfile.write('# objective {}\n'.format(self.objective))
            outfile.write('# elapsed time {}\n'.format(self._elapsed))
            writer = csv.writer(outfile, delimiter=self.delimiter)
            writer.writerow(['x_id', 'x_val'])
            for k, v in sorted(self._x_sol.items()):
                writer.writerow([k, v])

    @property
    def objective(self):
        return self._obj

    @property
    def time(self):
        return self._elapsed

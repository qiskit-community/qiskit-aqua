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
"""
The Variational Quantum Algorithm Base Class. This class can be used an interface for working with Variation Quantum
Algorithms, such as VQE, QAOA, or VSVM, and also provides helper utilities for implementing new variational algorithms.
Writing a new variational algorithm is a simple as extending this class, implementing a cost function for the new
algorithm to pass to the optimizer, and running the find_minimum() function below to begin the optimization.
Alternatively, all of the functions below can be overridden to opt-out of this infrastructure but still meet the
interface requirements.

"""

import time
import logging
import numpy as np

from qiskit import ClassicalRegister
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import find_regs_by_name
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class VQAlgorithm(QuantumAlgorithm):
    """
    The Variational Quantum Algorithm Base Class.
    """

    def __init__(self,
                 var_form=None,
                 optimizer=None,
                 cost_fn=None,
                 initial_point=None,
                 batch_mode=False,
                 callback=None):
        super().__init__()
        self._var_form = var_form
        self._optimizer = optimizer
        self._cost_fn = cost_fn
        self._initial_point = initial_point
        self._optimizer.set_batch_mode(batch_mode)
        self._ret = {}
        self._eval_count = 0
        self._eval_time = 0
        self._callback = callback

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running VQAlgorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running VQAlgorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running VQAlgorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc, decimals=16)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    def find_minimum(self, initial_point=None):
        """Optimize to find the minimum cost value.

        Returns:
            Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError:

        """
        self._eval_count = 0
        initial_point = initial_point if initial_point is not None else self._initial_point

        nparms = self._var_form.num_parameters
        bounds = self._var_form.parameter_bounds

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError('Initial point size {} and parameter size {} mismatch'.format(len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError('Variational form bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not self._optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if self._optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not self._optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if self._optimizer.is_initial_point_required:
                low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                initial_point = self.random.uniform(low, high)

        start = time.time()
        logger.info('Starting optimizer.\nbounds={}\ninitial point={}'.format(bounds, initial_point))
        opt_params, opt_val, num_optimizer_evals = self._optimizer.optimize(self._var_form.num_parameters,
                                                                            self._cost_fn,
                                                                            variable_bounds=bounds,
                                                                            initial_point=initial_point)
        if num_optimizer_evals is not None:
            self._eval_count = self._eval_count if self._eval_count >= num_optimizer_evals else num_optimizer_evals
        self._eval_time = time.time() - start
        logger.info('Optimization complete in {} seconds.\nFound opt_params {} in {} evals'.format(
            self._eval_time, opt_params, self._eval_count))

        self._ret['min_val'] = opt_val
        self._ret['opt_params'] = opt_params
        self._ret['eval_count'] = self._eval_count
        self._ret['eval_time'] = self._eval_time

        return opt_val, opt_params

    # Helper function to get probability vectors for a set of params
    def get_prob_vector_for_params(self, construct_circuit_fn, params_s,
                                   quantum_instance, construct_circuit_args=None):
        circuits = []
        for params in params_s:
            circuit = construct_circuit_fn(params, **construct_circuit_args)
            circuits.append(circuit)
        results = quantum_instance.execute(circuits)

        probs_s = []
        for circuit in circuits:
            if quantum_instance.is_statevector:
                sv = results.get_statevector(circuit)
                probs = np.real(sv * np.conj(sv))
                probs_s.append(probs)
            else:
                counts = results.get_counts(circuit)
                probs_s.append(self.get_probabilities_for_counts(counts))
        return np.array(probs_s)

    def get_probabilities_for_counts(self, counts):
        shots = sum(counts.values())
        states = int(2 ** len(list(counts.keys())[0]))
        probs = np.zeros(states)
        for k, v in counts.items():
            probs[int(k, 2)] = v / shots
        return probs

    @property
    def initial_point(self):
        return self._initial_point

    @initial_point.setter
    def initial_point(self, new_value):
        self._initial_point = new_value

    @property
    def optimal_params(self):
        return self._ret['opt_params']

    @property
    def var_form(self):
        return self._var_form

    @var_form.setter
    def var_form(self, new_value):
        self._var_form = new_value

    @property
    def optimizer(self):
        return self._optimizer

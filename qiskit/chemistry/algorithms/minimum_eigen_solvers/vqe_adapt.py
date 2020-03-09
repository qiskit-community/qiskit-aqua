# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
An adaptive VQE implementation.
"""

from typing import Optional, List
import logging
import warnings
import re
import numpy as np

from qiskit import ClassicalRegister
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQAlgorithm, VQE, VQEResult
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator, WeightedPauliOperator
from qiskit.aqua.utils.backend_utils import is_aer_statevector_backend
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.validation import validate_min

logger = logging.getLogger(__name__)


class VQEAdapt(VQAlgorithm):
    """
    The Adaptive VQE algorithm.

    See https://arxiv.org/abs/1812.11173
    """

    # TODO make re-usable, implement MinimumEignesolver interface
    def __init__(self, operator: BaseOperator,
                 var_form_base: VariationalForm, optimizer: Optimizer,
                 initial_point: Optional[np.ndarray] = None,
                 excitation_pool: Optional[List[WeightedPauliOperator]] = None,
                 threshold: float = 1e-5,
                 delta: float = 1, max_evals_grouped: int = 1,
                 aux_operators: Optional[List[BaseOperator]] = None) -> None:
        """
        Args:
            operator: Qubit operator
            var_form_base: base parameterized variational form
            optimizer: the classical optimizer algorithm
            initial_point: optimizer initial point
            excitation_pool: list of excitation operators
            threshold: absolute threshold value for gradients, has a min. value of 1e-15.
            delta: finite difference step size for gradient computation,
                    has a min. value of 1e-5.
            max_evals_grouped: max number of evaluations performed simultaneously
            aux_operators: Auxiliary operators to be evaluated
                                                at each eigenvalue

        Raises:
            ValueError: if var_form_base is not an instance of UCCSD.
            See also: qiskit/chemistry/components/variational_forms/uccsd_adapt.py
        """
        validate_min('threshold', threshold, 1e-15)
        validate_min('delta', delta, 1e-5)
        super().__init__(var_form=var_form_base,
                         optimizer=optimizer,
                         initial_point=initial_point)
        self._use_simulator_snapshot_mode = None
        self._ret = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        if initial_point is None:
            self._initial_point = var_form_base.preferred_init_points
        self._operator = operator
        if not isinstance(var_form_base, UCCSD):
            raise ValueError("var_form_base has to be an instance of UCCSD.")
        self._var_form_base = var_form_base
        self._var_form_base.manage_hopping_operators()
        self._excitation_pool = self._var_form_base.excitation_pool \
            if excitation_pool is None else excitation_pool
        self._threshold = threshold
        self._delta = delta
        self._aux_operators = []
        if aux_operators is not None:
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            for aux_op in aux_operators:
                self._aux_operators.append(aux_op)

    def _compute_gradients(self, excitation_pool, theta, delta,
                           var_form, operator, optimizer):
        """
        Computes the gradients for all available excitation operators.

        Args:
            excitation_pool (list): pool of excitation operators
            theta (list): list of (up to now) optimal parameters
            delta (float): finite difference step size (for gradient computation)
            var_form (VariationalForm): current variational form
            operator (BaseOperator): system Hamiltonian
            optimizer (Optimizer): classical optimizer algorithm

        Returns:
            list: List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        for exc in excitation_pool:
            # push next excitation to variational form
            var_form.push_hopping_operator(exc)
            # construct auxiliary VQE instance
            vqe = VQE(operator, var_form, optimizer)
            vqe.quantum_instance = self.quantum_instance
            vqe._operator = vqe._config_the_best_mode(operator, self.quantum_instance.backend)
            vqe._use_simulator_snapshot_mode = self._use_simulator_snapshot_mode
            # evaluate energies
            parameter_sets = theta + [-delta] + theta + [delta]
            energy_results = vqe._energy_evaluation(np.asarray(parameter_sets))
            # compute gradient
            gradient = (energy_results[0] - energy_results[1]) / (2*delta)
            res.append((np.abs(gradient), exc))
            # pop excitation from variational form
            var_form.pop_hopping_operator()

        return res

    def _run(self) -> 'VQEAdaptResult':
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        self._ret = {}  # TODO should be eliminated
        self._operator = VQE._config_the_best_mode(self, self._operator,
                                                   self._quantum_instance.backend)
        self._use_simulator_snapshot_mode = \
            is_aer_statevector_backend(self._quantum_instance.backend) \
            and isinstance(self._operator, (WeightedPauliOperator, TPBGroupedWeightedPauliOperator))
        self._quantum_instance.circuit_summary = True

        cycle_regex = re.compile(r'(.+)( \1)+')
        # reg-ex explanation:
        # 1. (.+) will match at least one number and try to match as many as possible
        # 2. the match of this part is placed into capture group 1
        # 3. ( \1)+ will match a space followed by the contents of capture group 1
        # -> this results in any number of repeating numbers being detected

        threshold_satisfied = False
        alternating_sequence = False
        prev_op_indices = []
        theta = []
        max_grad = ()
        iteration = 0
        while not threshold_satisfied and not alternating_sequence:
            iteration += 1
            logger.info('--- Iteration #%s ---', str(iteration))
            # compute gradients
            cur_grads = self._compute_gradients(self._excitation_pool, theta, self._delta,
                                                self._var_form_base, self._operator,
                                                self._optimizer)
            # pick maximum gradient
            max_grad_index, max_grad = max(enumerate(cur_grads),
                                           key=lambda item: np.abs(item[1][0]))
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            gradlog = "\nGradients in iteration #{}".format(str(iteration))
            gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
            for i, grad in enumerate(cur_grads):
                gradlog += '\n{}: {}: {}'.format(str(i), str(grad[1]), str(grad[0]))
                if grad[1] == max_grad[1]:
                    gradlog += '\t(*)'
            logger.info(gradlog)
            if np.abs(max_grad[0]) < self._threshold:
                logger.info("Adaptive VQE terminated succesfully with a final maximum gradient: %s",
                            str(np.abs(max_grad[0])))
                threshold_satisfied = True
                break
            # check indices of picked gradients for cycles
            if cycle_regex.search(' '.join(map(str, prev_op_indices))) is not None:
                logger.info("Alternating sequence found. Finishing.")
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                alternating_sequence = True
                break
            # add new excitation to self._var_form_base
            self._var_form_base.push_hopping_operator(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer,
                            initial_point=theta)
            vqe_result = algorithm.run(self._quantum_instance)
            self._ret['opt_params'] = vqe_result.optimal_point
            theta = vqe_result.optimal_point.tolist()
        # once finished evaluate auxiliary operators if any
        if self._aux_operators is not None and self._aux_operators:
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer,
                            initial_point=theta, aux_operators=self._aux_operators)
            vqe_result = algorithm.run(self._quantum_instance)
            self._ret['opt_params'] = vqe_result.optimal_point

        if threshold_satisfied:
            finishing_criterion = 'Threshold converged'
        elif alternating_sequence:
            finishing_criterion = 'Aborted due to cyclicity'
        else:
            raise AquaError('The algorithm finished due to an unforeseen reason!')

        # extend VQE returned information with additional outputs
        result = VQEAdaptResult()
        result.combine(vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.finishing_criterion = finishing_criterion

        logger.info('The final energy is: %s', str(result.optimal_value.real))
        return result

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form_base.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            tmp_cache = self._quantum_instance.circuit_cache
            self._quantum_instance._circuit_cache = None
            ret = self._quantum_instance.execute(qc)
            self._quantum_instance._circuit_cache = tmp_cache
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']


class VQEAdaptResult(VQEResult):
    """ VQE Result."""

    @property
    def num_iterations(self) -> int:
        """ Returns number of iterations """
        return self.get('num_iterations')

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """ Sets number of iterations """
        self.data['num_iterations'] = value

    @property
    def final_max_gradient(self) -> float:
        """ Returns final maximum gradient """
        return self.get('final_max_gradient')

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """ Sets final maximum gradient """
        self.data['final_max_gradient'] = value

    @property
    def finishing_criterion(self) -> str:
        """ Returns finishing criterion """
        return self.get('finishing criterion')

    @finishing_criterion.setter
    def finishing_criterion(self, value: str) -> None:
        """ Sets finishing criterion """
        self.data['finishing_criterion'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'final_max_grad':
            warnings.warn('final_max_grad deprecated, use final_max_gradient property.',
                          DeprecationWarning)
            return super().__getitem__('final_max_gradient')

        return super().__getitem__(key)

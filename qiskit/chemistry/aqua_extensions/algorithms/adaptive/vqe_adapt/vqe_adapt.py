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

"""
An adaptive VQE implementation.

See https://arxiv.org/abs/1812.11173
"""

import logging
import re
import numpy as np

from qiskit import ClassicalRegister
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm
from qiskit.aqua.algorithms.adaptive.vqe.vqe import VQE
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.aqua.operators import TPBGroupedWeightedPauliOperator, WeightedPauliOperator
from qiskit.aqua.utils.backend_utils import is_aer_statevector_backend

logger = logging.getLogger(__name__)


class VQEAdapt(VQAlgorithm):
    """
    An adaptive VQE implementation.

    See https://arxiv.org/abs/1812.11173
    """

    CONFIGURATION = {
        'name': 'VQEAdapt',
        'description': 'Adaptive VQE Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'vqe_adapt_schema',
            'type': 'object',
            'properties': {
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'excitation_pool': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "WeightedPauliOperator"
                    },
                    'default': None
                },
                'threshold': {
                    'type': 'float',
                    'minimum': 1e-15,  # limited by floating point precision
                    'default': 1e-5

                },
                'delta': {
                    'type': 'float',
                    'minimum': 1e-5,
                    'default': 1
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'L_BFGS_B'
                },
            },
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'UCCSD'
                },
            },
        ],
    }

    def __init__(self, operator, var_form_base, optimizer, initial_point=None,
                 excitation_pool=None, threshold=1e-5, delta=1,
                 max_evals_grouped=1, aux_operators=None):
        """Constructor.

        Args:
            operator (BaseOperator): Qubit operator
            var_form_base (VariationalForm): base parametrized variational form
            optimizer (Optimizer): the classical optimizer algorithm
            initial_point (numpy.ndarray): optimizer initial point
            excitation_pool (list[WeightedPauliOperator]): list of excitation operators
            threshold (double): absolute threshold value for gradients
            delta (float): finite difference step size for gradient computation
            max_evals_grouped (int): max number of evaluations performed simultaneously
            aux_operators (list[BaseOperator]): Auxiliary operators to be evaluated
                                                at each eigenvalue

        Raises:
            ValueError: if var_form_base is not an instance of UCCSD.
            See also: qiskit/chemistry/aqua_extensions/components/variational_forms/uccsd_adapt.py
        """
        super().__init__(var_form=var_form_base,
                         optimizer=optimizer,
                         initial_point=initial_point)
        self._use_simulator_operator_mode = None
        self._ret = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        if initial_point is None:
            self._initial_point = var_form_base.preferred_init_points
        self._operator = operator
        if not isinstance(var_form_base, UCCSD):
            raise ValueError("var_form_base has to be an instance of UCCSD.")
        self._var_form_base = var_form_base
        self._var_form_base._manage_hopping_operators()
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

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance

        Returns:
            VQEAdapt: VQEAdapt object
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        vqe_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        initial_point = vqe_params.get('initial_point')
        excitation_pool = vqe_params.get('excitation_pool')
        threshold = vqe_params.get('threshold')
        delta = vqe_params.get('delta')
        max_evals_grouped = vqe_params.get('max_evals_grouped')

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, var_form, optimizer, excitation_pool=excitation_pool,
                   initial_point=initial_point, threshold=threshold, delta=delta,
                   max_evals_grouped=max_evals_grouped, aux_operators=algo_input.aux_ops)

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
            var_form._push_hopping_operator(exc)
            # construct auxiliary VQE instance
            vqe = VQE(operator, var_form, optimizer)
            vqe._quantum_instance = self._quantum_instance
            vqe._use_simulator_operator_mode = self._use_simulator_operator_mode
            # evaluate energies
            parameter_sets = theta + [-delta] + theta + [delta]
            energy_results = vqe._energy_evaluation(np.asarray(parameter_sets))
            # compute gradient
            gradient = (energy_results[0] - energy_results[1]) / (2*delta)
            res.append((np.abs(gradient), exc))
            # pop excitation from variational form
            var_form._pop_hopping_operator()

        return res

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        self._operator = VQE._config_the_best_mode(self, self._operator,
                                                   self._quantum_instance.backend)
        self._use_simulator_operator_mode = \
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
            self._var_form_base._push_hopping_operator(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer,
                            initial_point=theta)
            self._ret = algorithm.run(self._quantum_instance)
            theta = self._ret['opt_params'].tolist()
        # once finished evaluate auxiliary operators if any
        if self._aux_operators is not None and self._aux_operators:
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer,
                            initial_point=theta, aux_operators=self._aux_operators)
            self._ret = algorithm.run(self._quantum_instance)
        # extend VQE returned information with additional outputs
        logger.info('The final energy is: %s', str(self._ret['energy']))
        self._ret['num_iterations'] = iteration
        self._ret['final_max_grad'] = max_grad[0]
        if threshold_satisfied:
            self._ret['finishing_criterion'] = 'threshold_converged'
        elif alternating_sequence:
            self._ret['finishing_criterion'] = 'aborted_due_to_cyclicity'
        else:
            raise AquaError('The algorithm finished due to an unforeseen reason!')
        return self._ret

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

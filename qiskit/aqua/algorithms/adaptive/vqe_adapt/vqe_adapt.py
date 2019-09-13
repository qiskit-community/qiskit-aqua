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
import functools
import warnings

import numpy as np

from qiskit.aqua import Operator
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm
from qiskit.aqua.algorithms.adaptive.vqe.vqe import VQE
from qiskit.chemistry.aqua_extensions.components.variational_forms.ucc import UCC
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator, WeightedPauliOperator,
                                   MatrixOperator, op_converter)
from qiskit.aqua.utils.backend_utils import is_aer_statevector_backend, is_statevector_backend

logger = logging.getLogger(__name__)


class VQEAdapt(VQAlgorithm):
    """
    An adaptive VQE implementation.

    See https://arxiv.org/abs/1812.11173
    """

    CONFIGURATION = {
        'name': 'VQEAdapt',
        'description': 'Adaptive VQE Algorithm'
    }

    def __init__(self, operator, var_form_base, threshold,
                 optimizer, excitation_pool, initial_point=None):
        """Constructor.

        Args:
            operator (BaseOperator): Qubit operator
            var_form_base (VariationalForm): base parametrized variational form
            threshold (double): absolute threshold value for gradients
            optimizer (Optimizer): the classical optimizer algorithm
            initial_point (numpy.ndarray): optimizer initial point
        """
        super().__init__(var_form=var_form_base,
                         optimizer=optimizer,
                         initial_point=initial_point)
        if initial_point is None:
            self._initial_point = var_form_base.preferred_init_points
        if isinstance(operator, Operator):
            warnings.warn("operator should be type of BaseOperator, Operator type is deprecated and "
                          "it will be removed after 0.6.", DeprecationWarning)
            operator = op_converter.to_weighted_pauli_operator(operator)
        self._operator = operator
        if not isinstance(var_form_base, UCC):
            warnings.warn("var_form_base has to be an instance of UCC.")
            return 1
        self._var_form_base = var_form_base
        self._excitation_pool = excitation_pool
        self._threshold = threshold

    def _compute_gradients(self, var_form, delta, optimizer, theta, excitation_pool, operator):

        res = []

        for exc in excitation_pool:
            var_form._append_hopping_operator(exc)

            vqe = VQE(operator, var_form, optimizer)

            if theta == []:
                params_minus = [-delta]
            else:
                params_minus = theta + [-delta]
            circuit_minus = vqe.construct_circuit(params_minus,
                                                  statevector_mode=self._quantum_instance.is_statevector,
                                                  use_simulator_operator_mode=self._use_simulator_operator_mode,
                                                  circuit_name_prefix='minus')

            to_be_simulated_circuit_minus = functools.reduce(lambda x, y: x + y, circuit_minus)

            if self._use_simulator_operator_mode:
                extra_args = {'expectation': {
                    'params': [self._operator.aer_paulis],
                    'num_qubits': self._operator.num_qubits}
                }
            else:
                extra_args = {}
            result_minus = self._quantum_instance.execute(to_be_simulated_circuit_minus, **extra_args)

            mean_minus, std_minus = self._operator.evaluate_with_result(
                    result=result_minus, statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_operator_mode=self._use_simulator_operator_mode,
                    circuit_name_prefix='minus')

            if theta == []:
                params_plus = [delta]
            else:
                params_plus = theta + [delta]
            circuit_plus = vqe.construct_circuit(params_plus,
                                                 statevector_mode=self._quantum_instance.is_statevector,
                                                 use_simulator_operator_mode=self._use_simulator_operator_mode,
                                                 circuit_name_prefix='plus')
            to_be_simulated_circuit_plus = functools.reduce(lambda x, y: x + y, circuit_plus)
            if self._use_simulator_operator_mode:
                extra_args = {'expectation': {
                    'params': [self._operator.aer_paulis],
                    'num_qubits': self._operator.num_qubits}
                }
            else:
                extra_args = {}
            result_plus = self._quantum_instance.execute(to_be_simulated_circuit_plus, **extra_args)

            mean_plus, std_plus = self._operator.evaluate_with_result(
                result=result_plus, statevector_mode=self._quantum_instance.is_statevector,
                use_simulator_operator_mode=self._use_simulator_operator_mode,
                circuit_name_prefix='plus')

            res.append(((mean_minus-mean_plus)/(2*delta), exc))

            var_form._pop_hopping_operator()

        return res

    def _config_the_best_mode(self, operator, backend):

        if not isinstance(operator, (WeightedPauliOperator, MatrixOperator, TPBGroupedWeightedPauliOperator)):
            logger.debug("Unrecognized operator type, skip auto conversion.")
            return operator

        ret_op = operator
        if not is_statevector_backend(backend):  # assume qasm, should use grouped paulis.
            if isinstance(operator, (WeightedPauliOperator, MatrixOperator)):
                logger.debug("When running with Qasm simulator, grouped pauli can save number of measurements. "
                             "We convert the operator into grouped ones.")
                ret_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    operator, TPBGroupedWeightedPauliOperator.sorted_grouping)
        else:
            if not is_aer_statevector_backend(backend):
                if not isinstance(operator, MatrixOperator):
                    logger.info("When running with non-Aer statevector simulator, represent operator as a matrix could "
                                "achieve the better performance. We convert the operator to matrix.")
                    ret_op = op_converter.to_matrix_operator(operator)
            else:
                if not isinstance(operator, WeightedPauliOperator):
                    logger.info("When running with Aer statevector simulator, represent operator as weighted paulis could "
                                "achieve the better performance. We convert the operator to weighted paulis.")
                    ret_op = op_converter.to_weighted_pauli_operator(operator)
        return ret_op

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        self._operator = self._config_the_best_mode(self._operator, self._quantum_instance.backend)
        self._use_simulator_operator_mode = \
            is_aer_statevector_backend(self._quantum_instance.backend) \
            and isinstance(self._operator, (WeightedPauliOperator, TPBGroupedWeightedPauliOperator))
        self._quantum_instance.circuit_summary = True

        threshold_satisfied = False
        theta = []
        iteration = 0
        while not threshold_satisfied:
            iteration += 1
            # compute gradients
            cur_grads = self._compute_gradients(self._var_form_base, 0.0001, self._optimizer, theta,
                                                self._excitation_pool, self._operator)
            # pick maximum gradients and choose that excitation
            max_grad = max(cur_grads, key=lambda item: np.abs(item[0]))
            if np.abs(max_grad[0]) < self._threshold:
                threshold_satisfied = True
                break
            # add new excitation to self._var_form_base
            self._var_form_base._append_hopping_operator(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer, initial_point=theta)
            self._ret = algorithm.run(self._quantum_instance)
            theta = self._ret['opt_params'].tolist()
            print('Iteration #' + str(iteration) + ': E = ' + str(self._ret['energy']))
        return self._ret

    def get_optimal_cost(self):
        pass

    def get_optimal_circuit(self):
        pass

    def get_optimal_vector(self):
        pass

    def optimal_params(self):
        pass

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
import warnings

from qiskit.aqua import Operator
from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm
from qiskit.aqua.algorithms.adaptive.vqe.vqe import VQE
from qiskit.chemistry.aqua_extensions.components.variational_forms.ucc import UCC
from qiskit.aqua.operators import op_converter

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
            '$schema': 'http://json-schema.org/schema#',
            'id': 'vqe_schema',
            'type': 'object',
            'properties': {
                'operator_mode': {
                    'type': ['string', 'null'],
                    'default': None,
                    'enum': ['matrix', 'paulis', 'grouped_paulis', None]
                },
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
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
            {'pluggable_type': 'optimizer',
             'default': {
                     'name': 'L_BFGS_B'
                }
             },
            {'pluggable_type': 'variational_form',
             'default': {
                     'name': 'RYRZ'
                }
             },
        ],
    }

    def __init__(self, operator, var_form_base, threshold,
                 optimizer, initial_point=None):
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
        self._excitation_pool = self._var_form_base._hopping_ops
        self._threshold = threshold

    def _compute_gradients(self):
        """
        # TODO
        """

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        threshold_satisfied = False
        theta = []
        while not threshold_satisfied:
            # compute gradients
            cur_grads = self._compute_gradients()
            # pick maximum gradients and choose that excitation
            max_grad = max(cur_grads, key=lambda item: item[1])
            if max_grad[1] < self._threshold:
                threshold_satisfied = True
                break
            # add new excitation to self._var_form_base
            self._var_form_base = self._var_form_base  # .var_form_function(max_grad[0])
            theta.append(0.0)
            # run VQE on current Ansatz
            algorithm = VQE(self._operator, self._var_form_base, self._optimizer, initial_point=theta)
            self._ret = algorithm.run(self._quantum_instance)
            theta = self._ret['eigvals']
        return self._ret

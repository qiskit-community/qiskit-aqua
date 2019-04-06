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

import logging
import numpy as np
import math

from qiskit.aqua.components.gradients import Gradient
from qiskit.aqua.algorithms.adaptive.qsvm import (cost_estimate, return_probabilities)


logger = logging.getLogger(__name__)


class CircuitsGradient(Gradient):
    CONFIGURATION = {
        'name': 'CircuitsGradient',
        'description': 'CircuitsGradient',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'gradient_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, params=None):
        super().__init__()

    def _get_cost(self, theta):  #todo eval count logger
        """Compute the objective/cost value for each data point theta.
        Internally, we first construct the circuits by concatenating the feature_map circuits with the var_form circuit. Next we evaluate the circuits to compute the probability distribution. Lastly, we compute the cost value based on the probability distribution computed and the labels.
        Args:
            theta (ndarray): 1-d array that represents the parameter values
        Returns:
            float: the objective/cost value
        """
        circuits_to_eval = {}
        circuit_id = 0

        num_theta_sets = len(theta) // self.var_form.num_parameters
        theta_sets = np.split(theta, num_theta_sets)

        for theta in theta_sets:
            for circuit in self.circuits:
                qr = circuit.qregs[0]
                cr = circuit.cregs[0]
                if self.quantum_instance.is_statevector:
                    circuits_to_eval[circuit_id] = circuit + self.var_form.construct_circuit(theta, qr)
                else:
                    circuit += self.var_form.construct_circuit(theta, qr)
                    circuit.barrier(qr)
                    circuit.measure(qr, cr)
                    circuits_to_eval[circuit_id] = circuit
                circuit_id += 1

        results = self.quantum_instance.execute(list(circuits_to_eval.values()))

        circuit_id = 0
        predicted_probs = []
        predicted_labels = []
        for theta in theta_sets:
            counts = []
            for circuit in self.circuits:
                if self.quantum_instance.is_statevector:
                    temp = results.get_statevector(circuits_to_eval[circuit_id])
                    outcome_vector = (temp * temp.conj()).real
                    # convert outcome_vector to outcome_dict, where key is a basis state and value is the count.
                    # Note: the count can be scaled linearly, i.e., it does not have to be an integer.
                    outcome_dict = {}
                    bitstringsize = int(math.log2(len(outcome_vector)))
                    for i in range(len(outcome_vector)):
                        bitstr_i = format(i, '0' + str(bitstringsize) +'b')
                        outcome_dict[bitstr_i] = outcome_vector[i]
                else:
                    outcome_dict = results.get_counts(circuits_to_eval[circuit_id])

                counts.append(outcome_dict)
                circuit_id += 1

            probs = return_probabilities(counts, self.num_classes)
            predicted_probs.append(probs)
            predicted_labels.append(np.argmax(probs, axis=1))

        if len(predicted_probs) == 1:
            predicted_probs = predicted_probs[0]
        if len(predicted_labels) == 1:
            predicted_labels = predicted_labels[0]


        total_cost = []
        if not isinstance(predicted_probs, list):
            predicted_probs = [predicted_probs]
        for i in range(len(predicted_probs)):
            curr_cost = cost_estimate(predicted_probs[i], self.labels)
            total_cost.append(curr_cost)
        return total_cost if len(total_cost) > 1 else total_cost[0]

    def get_gradient_function(self, var_form, circuits, quantum_instance, num_classes, labels):
        """Return the gradient function for computing the gradient at a point
        Args:
            var_form (VariationalForm): the variational form instance
            circuits (list): list of circuits that encode the n data points
            quantum_instance (QuantumInstance): QuantumInstance for the execution
            num_classes (int): number of classes in the classification problem
            labels (ndarray): 1-d array that shows the classification labels for the n data points
        Returns:
            func: the gradient function for computing the gradient at a point
        """
        self.var_form = var_form
        self.circuits = circuits
        self.quantum_instance = quantum_instance
        self.num_classes = num_classes
        self.labels = labels
        self.objective_function = self._get_cost

        def gradient_num_diff(x_center):
            """
            We compute the gradient with the numeric differentiation  around the point x_center.
            Args:
                x_center (ndarray): point around which we compute the gradient
            Returns:
                grad: the gradient computed
            """
            epsilon = 1e-8
            forig = self.objective_function(*((x_center,)))
            grad = []
            ei = np.zeros((len(x_center),), float)
            todos = []
            for k in range(len(x_center)):
                ei[k] = 1.0
                d = epsilon * ei
                todos.append(x_center + d)
                deriv = (self.objective_function(x_center + d) - forig)/epsilon
                grad.append(deriv)
                ei[k] = 0.0
            return np.array(grad)

        return gradient_num_diff












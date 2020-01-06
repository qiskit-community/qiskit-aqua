
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
The Iterative Quantum Amplitude Estimation Algorithm.
"""


import logging
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError

from .ae_algorithm import AmplitudeEstimationAlgorithm

logger = logging.getLogger(__name__)


class SimplifiedAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    r"""
    Aar: (1 - eps) sqrt(a) < sqrt(\hat a) < (1 + eps) sqrt(a)
        => (1 - 2eps + eps^2) a < \hat a < (1 + 2eps + eps^2) a
        => (1 - 2eps - eps^2) a < \hat a < (1 + 2eps + eps^2) a

    Want: (1 - e) a < \hat a < (1 + e) a

    Thus, e = 2eps + eps^2 = eps(2 + eps), or eps = ...
    """

    def __init__(self, epsilon, delta, a_factory=None, q_factory=None, i_objective=None):
        """
        todo

        Args:
            epsilon (float): target precision for estimation target `a`
            delta (float): confidence level, the target probability is 1 - alpha
            a_factory (CircuitFactory): the A operator, specifying the QAE problem
            q_factory (CircuitFactory): the Q operator (Grover operator), constructed from the
                A operator
            i_objective (int): index of the objective qubit, that marks the 'good/bad' states
        """
        # self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # store parameters
        self._transformed_epsilon = epsilon  # transform to aaronsons eps
        self._delta = delta

        # results dictionary
        self._ret = {}

    @property
    def precision(self):
        """
        Returns the target precision `epsilon` of the algorithm

        Returns:
            float: target precision
        """
        return self._epsilon

    @precision.setter
    def precision(self, epsilon):
        """
        Set the target precision of the algorithm.

        Args:
            epsilon (float): target precision for estimation target a
        """
        self._epsilon = epsilon
        self._transformed_epsilon = epsilon  # TODO transform correctly

    def _find_next_r(self, k, upper_half_circle, theta_interval, min_ratio=2):
        """
        todo
        """
        pass

    def construct_circuit(self, k, measurement=False):
        r"""
        Construct the circuit Q^k A \|0>, with the A operator specifying the QAE problem and
        the Grover operator Q.

        Args:
            k (int): the power of Q operator
            measurement (bool): boolean flag to indicate if measurements should be included in the
                circuits

        Returns:
            QuantumCircuit: the circuit Q^k A \|0>
        """
        # set up circuit
        q = QuantumRegister(self.a_factory.num_target_qubits, 'q')
        circuit = QuantumCircuit(q, name='circuit')

        # get number of ancillas and add register if needed
        num_ancillas = np.maximum(self.a_factory.required_ancillas(),
                                  self.q_factory.required_ancillas())

        q_aux = None
        # pylint: disable=comparison-with-callable
        if num_ancillas > 0:
            q_aux = QuantumRegister(num_ancillas, 'aux')
            circuit.add_register(q_aux)

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(1)
            circuit.add_register(c)

        # add A operator
        self.a_factory.build(circuit, q, q_aux)

        # add Q^k
        self.q_factory.build_power(circuit, q, k, q_aux)

        # add optional measurement
        if measurement:
            circuit.measure(q[self.i_objective], c[0])

        return circuit

    def _run(self):
        # check if A factory has been set
        if self.a_factory is None:
            raise AquaError("a_factory must be set!")

        # for statevector we can directly return the probability to measure 1
        # note, that no iterations here are necessary
        if self._quantum_instance.is_statevector:
            # simulate circuit
            circuit = self.construct_circuit(k=0, measurement=False)
            ret = self._quantum_instance.execute(circuit)
            num_oracle_queries = 1

            # get statevector
            statevector = ret.get_statevector(circuit)

            # calculate the probability of measuring '1'
            num_qubits = self.a_factory.num_target_qubits

            # sum over all amplitudes where the objective qubit is 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                if ('{:0%db}' % num_qubits).format(i)[-(1 + self.i_objective)] == '1':
                    prob = prob + np.abs(amplitude)**2

            # get the estimate and confidence intervals
            value = prob
            theta_min, theta_max = 2 * [np.arcsin(np.sqrt(np.sqrt(value) / 1000))]
            confint = 2 * [value]

        else:
            num_oracle_queries = 0

            # step 1
            num_samples = int(np.ceil(1e5 * np.log(120 / self._delta)))
            self._ret['num_samples_step1'] = num_samples
            self._ret['r_step1'] = []

            t = 0
            good_counts = 0
            while good_counts < num_samples / 3:
                # find the nearest odd integer
                r = np.ceil((12 / 11)**t)
                if r % 2 == 0:
                    r -= 1
                self._ret['r_step1'] += [r]

                # execute the circuit
                circuit = self.construct_circuit((r - 1) / 2, measurement=True)
                ret = self._quantum_instance.execute(circuit, shots=num_samples)
                num_oracle_queries += num_samples * np.maximum(r, 1)

                # get the counts
                counts = ret.get_counts(circuit)
                good_counts = counts['1']

                # increase t
                t += 1

            # step 2
            theta_min = 5 / 8 * (11 / 12)**(t + 1)
            theta_max = 5 / 8 * (11 / 12)**(t - 1)

            self._ret['num_samples_step2'] = []
            self._ret['r_step2'] = []

            t = 0
            while theta_max > (1 + self._transformed_epsilon / 5) * theta_min:
                num_samples = int(
                    np.ceil(1000 * np.log(100 / self._delta / self._transformed_epsilon * 0.9**t)))
                self._ret['num_samples_step2'] += [num_samples]

                # find next r
                r = self._find_next_r(theta_min, theta_max)  # constant time
                self._ret['r_step2'] += [r]

                # execute the circuit
                circuit = self.construct_circuit((r - 1) / 2, measurement=True)
                ret = self._quantum_instance.execute(circuit, shots=num_samples)
                num_oracle_queries += num_samples * np.maximum(r, 1)

                # get the counts
                counts = ret.get_counts(circuit)
                good_counts = counts['1']

                # update bounds
                theta_ratio = theta_max / theta_min - 1  # called gamma in Aaronson's paper
                if good_counts >= num_samples / 2:
                    theta_min = theta_max / (1 + 0.9 * theta_ratio)
                else:
                    theta_max = (1 + 0.9 * theta_ratio) * theta_min

                # increase t
                t += 1

            # get the estimate and confidence intervals
            value = (1000 * np.sin(theta_max))**2
            confint = [(1 - self._epsilon) * value, (1 + self._epsilon) * value]

        estimation = self.a_factory.value_to_estimation(value)
        mapped_confint = [self.a_factory.value_to_estimation(bound) for bound in confint]

        # update results dictionary
        self._ret['value'] = value
        self._ret['estimation'] = estimation
        self._ret['num_oracle_queries'] = num_oracle_queries
        self._ret['theta_interval'] = [theta_min, theta_max]
        self._ret['alpha'] = self._delta
        self._ret['confidence_interval'] = mapped_confint

        return self._ret

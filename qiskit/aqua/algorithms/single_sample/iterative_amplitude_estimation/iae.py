# TODO
# (c) 2019 IBM

"""
An iterative version of the Amplitude Estimation algorithm based on
https://arxiv.org/pdf/1904.10246.pdf
"""

import logging
# from collections import OrderedDict
import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
# from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.q_factory import QFactory
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.ci_utils import chi2_quantile, normal_quantile

logger = logging.getLogger(__name__)


class IterativeAmplitudeEstimation(QuantumAlgorithm):

    CONFIGURATION = {
        'name': 'IterativeAmplitudeEstimation',
        'description': 'Iterative Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'AmplitudeEstimation_schema',
            'type': 'object',
            'properties': {
                'num_eval_qubits': {
                    'type': 'integer',
                    'default': 5,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['uncertainty'],
        'depends': [
            {
                'pluggable_type': 'uncertainty_problem',
                'default': {
                    'name': 'EuropeanCallDelta'
                }
            },
        ],
    }

    def __init__(self, num_iterations, a_factory,
                 q_factory=None, rotations=None, i_objective=None):
        """
        Constructor.

        Args:
            num_iterations (int): number of iterations in the algorithm
            a_factory (CircuitFactory): the CircuitFactory subclass object
                                        representing the problem unitary
            q_factory (CircuitFactory): the CircuitFactory subclass object
                                        representing an amplitude estimation
                                        sample (based on a_factory)
            rotations (iterable of ints): The number of times the operator Q
                                          is applied in each iteration,
                                          overwrites num_iterations
        """
        super().__init__()

        # get/construct A/Q operator
        self.a_factory = a_factory
        if q_factory is None:
            if i_objective is None:
                i_objective = self.a_factory.num_target_qubits - 1
            self.q_factory = QFactory(a_factory, i_objective)
        else:
            self.q_factory = q_factory

        # get parameters
        self._num_iterations = num_iterations
        if rotations is None:
            self._rotations = 2**np.arange(num_iterations)
        else:
            self._rotations = rotations

        # Store likelihood functions of single experiments here
        self._likelihoods = []

        # Store the number of good counts, needed for observed Fisher info
        self._good_counts = []

        # Results dictionary
        self._ret = {}

        # determine number of ancillas
        self._num_ancillas = self.q_factory.required_ancillas_controlled()
        self._num_qubits = self.a_factory.num_target_qubits + \
            + self._num_ancillas

    def get_single_likelihood(self, good, total, num_rotations):
        """
        @brief Likelihood function for a Amplitude Amplification experiment
        @param good Number of times we measured |1>
        @param total Number of times we measured
        @param num_rotations The amount of times we applied Q
        @return Function handle for the likelihood (*not* log-likelihood!)
        """
        def likelihood(theta):
            L = (np.sin((2 * num_rotations + 1) * theta)**2)**good \
                * (np.cos((2 * num_rotations + 1) * theta)**2)**(total - good)
            return L

        return likelihood

    def construct_single_circuit(self, num_rotations, measurement=False):
        q = QuantumRegister(self.q_factory.num_target_qubits, name='q')
        self._state_register = q

        qc = QuantumCircuit(q)

        num_aux_qubits, aux = 0, None
        if self.a_factory is not None:
            num_aux_qubits = self.a_factory.required_ancillas()
        if self.q_factory is not None:
            num_aux_qubits = max(num_aux_qubits,
                                 self.q_factory.required_ancillas_controlled())

        if num_aux_qubits > 0:
            aux = QuantumRegister(num_aux_qubits, name='aux')
            qc.add_register(aux)

        self.a_factory.build(qc, q, aux)
        self.q_factory.build_power(qc, q, num_rotations, aux)

        if measurement:
            q_ancilla = ClassicalRegister(self.q_factory.num_target_qubits,
                                          name='qa')
            qc.add_register(q_ancilla)
            # qc.barrier(a)
            qc.measure(q, q_ancilla)

        return qc

    def maximize(self, likelihood):
        # Should be this many numbers also for LR statistic later in self.ci!
        thetas = np.linspace(0, np.pi / 2, num=int(1e6))
        vals = np.array([likelihood(t) for t in thetas])
        vals = np.array([np.maximum(v, 1e-8) for v in vals])

        # Avoid double evaluation in likelihood ratio
        self._thetas_grid = thetas
        self._logliks_grid = np.log(vals)

        idx = np.argmax(vals)

        return thetas[idx]

    def get_mle(self):
        if len(self._likelihoods) == 0:
            raise AquaError("likelihoods empty, call the method `run` first")

        def likelihood(theta):
            return np.prod([lik(theta) for lik in self._likelihoods])

        return self.maximize(likelihood)

    def ci(self, alpha, kind="likelihood_ratio", plot=None):
        if kind == "likelihood_ratio":
            # Threshold defining confidence interval
            loglik_mle = np.max(self._logliks_grid)
            thres = loglik_mle - chi2_quantile(alpha) / 2

            # Which values are above the threshold?
            above_thres = self._thetas_grid[self._logliks_grid >= thres]

            # Get boundaries
            # since thetas_grid is sorted [0] == min, [1] == max
            ci_angle = np.array([above_thres[0], above_thres[-1]])
            ci = np.sin(ci_angle)**2

            if plot == "single":
                import matplotlib.pyplot as plt
                plt.title("Log likelihood for iterative Amplitude Estimation")
                plt.plot(np.sin(self._thetas_grid)**2,
                         self._logliks_grid,
                         label="log likelihood")
                plt.plot(self._ret['estimation'],
                         loglik_mle,
                         "ro",
                         label="Estimator")
                plt.axhline(y=thres, color="k", linestyle="--")
                [plt.axvline(x=bound, color="r", linestyle=":") for bound in ci]
                plt.xlabel("Value $a^*$")
                plt.ylabel("$\\log L(a^*)$")
                plt.legend(loc="best")
                plt.show()

            if plot == "joint":
                print("""
                      You called joint plots, this argument makes only sense if
                      you call first likelihood_ratio and then
                      likelihood_ratio_min
                      """)
                import matplotlib.pyplot as plt
                plt.plot(np.sin(self._thetas_grid)**2,
                         self._logliks_grid,
                         label="log likelihood")
                plt.plot(self._ret['estimation'],
                         loglik_mle,
                         "ro",
                         label="Estimator")
                plt.axhline(y=thres, color="k", linestyle="--")
                plt.axvline(x=ci[0], color="r", linestyle="-.",
                            label="outer bounds")
                plt.axvline(x=ci[1], color="r", linestyle="-.")

            return ci

        if kind == "likelihood_ratio_min":
            # Threshold defining confidence interval
            loglik_mle_idx = np.argmax(self._logliks_grid)
            loglik_mle = self._logliks_grid[loglik_mle_idx]
            thres = loglik_mle - chi2_quantile(alpha) / 2

            # Look for the first sign change starting from MLE
            diff = self._logliks_grid - thres
            ci_angle = []
            for direction in [-1, 1]:
                changed = False
                idx = loglik_mle_idx
                while not changed:
                    next = idx + direction
                    if diff[idx] * diff[next] < 0:
                        changed = True
                        ci_angle.append(self._thetas_grid[idx])
                    idx = next

            ci = np.sin(ci_angle)**2

            if plot == "single":
                import matplotlib.pyplot as plt
                plt.title("Log likelihood for iterative Amplitude Estimation")
                plt.plot(np.sin(self._thetas_grid)**2,
                         self._logliks_grid,
                         label="log likelihood")
                plt.plot(self._ret['estimation'],
                         loglik_mle,
                         "ro",
                         label="Estimator")
                plt.axhline(y=thres, color="k", linestyle="--")
                [plt.axvline(x=bound, color="orange", linestyle=":") for bound in ci]
                plt.xlabel("Value $a^*$")
                plt.ylabel("$\\log L(a^*)$")
                plt.legend(loc="best")
                plt.show()

            if plot == "joint":
                import matplotlib.pyplot as plt
                plt.axvline(x=ci[0], color="orange", linestyle=":",
                            label="inner bounds")
                plt.axvline(x=ci[1], color="orange", linestyle=":")
                plt.title("Log likelihood for iterative Amplitude Estimation")
                plt.xlabel("Value $a^*$")
                plt.ylabel("$\\log L(a^*)$")
                plt.legend(loc="best")
                plt.show()

            return ci

        if kind == "fisher":
            q = normal_quantile(alpha)
            est = self._ret['estimation']

            shots = sum(self._ret['counts'].values())
            fi = shots / (est * (1 - est)) * \
                sum((2 * nr + 1)**2 for nr in self._rotations)

            ci = est + np.array([-1, 1]) * q / np.sqrt(fi)
            return ci

        if kind == "observed_fisher":
            q = normal_quantile(alpha)
            angle = self._ret['angle']
            est = self._ret['estimation']

            shots = sum(self._ret['counts'].values())
            obs_fi = 0
            for nr, hk in zip(self._rotations, self._good_counts):
                mk = (2 * nr + 1)
                tan = np.tan(mk * angle)
                obs_fi += (2 * mk * (hk / tan - (shots - hk) * tan))**2

            ci = est + np.array([-1, 1]) * q / np.sqrt(obs_fi)
            return ci

        else:
            raise AquaError("confidence interval kind {} not implemented".format(kind))

    def last_qubit_is_one(self, i):
        n = self.a_factory._uncertainty_model.num_target_qubits
        return "{0:b}".format(i).rjust(n + 1, "0")[0] == "1"

    def _run(self):
        for num_rotations in self._rotations:
            if self._quantum_instance.is_statevector:
                qc = self.construct_single_circuit(num_rotations,
                                                   measurement=False)
                # run circuit on statevector simlator
                ret = self._quantum_instance.execute(qc)
                state_vector = np.asarray([ret.get_statevector(qc)])
                self._ret['statevector'] = state_vector

                # get all states where the last qubit is 1
                n = self.a_factory._uncertainty_model.num_target_qubits
                good_states = np.array([i for i in np.arange(
                    2**(n + 1)) if self.last_qubit_is_one(i)])

                # sum over all probabilities of these states
                amplitudes = np.real(state_vector.conj() * state_vector)[0]
                pr_good = np.sum(amplitudes[good_states])

                # get the counts
                good_counts = pr_good
                total_counts = 1

            else:
                # run circuit on QASM simulator
                qc = self.construct_single_circuit(num_rotations,
                                                   measurement=True)
                ret = self._quantum_instance.execute(qc)

                # get counts
                self._ret['counts'] = ret.get_counts()

                # sum all counts where last qubit is one
                good_counts = 0
                counts = ret.get_counts()
                for state, count in counts.items():
                    if state[0] == '1':
                        good_counts += count

                # normalise the counts, otherwise the maximum search
                # is numerically very difficult, as the values tend to 0
                total_counts = sum(ret.get_counts().values())
                good_counts /= total_counts
                total_counts = 1

            self._good_counts.append(good_counts)
            self._likelihoods.append(self.get_single_likelihood(good_counts,
                                                                total_counts,
                                                                num_rotations))

        self._ret['angle'] = self.get_mle()
        self._ret['value'] = np.sin(self._ret['angle'])**2
        self._ret['estimation'] = self.a_factory.value_to_estimation(
            self._ret['value'])

        return self._ret

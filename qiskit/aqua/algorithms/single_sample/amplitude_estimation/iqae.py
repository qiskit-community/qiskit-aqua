import logging
import numpy as np
import statsmodels
from statsmodels.stats.proportion import proportion_confint

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class

from .ae_base import (AmplitudeEstimationBase)

logger = logging.getLogger(__name__)

class IterativeAmplitudeEstimation(AmplitudeEstimationBase):
    CONFIGURATION = {
        'name': 'IterativeAmplitudeEstimation',
        'description': 'Iterative Amplitude Estimation',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'IterativeAmplitudeEstimation_schema',
            'type': 'object',
            'properties': {
                'precision': {
                    'type': 'integer',
                    'default': 10,
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

    def __init__(self, epsilon, alpha, method = 'beta', min_ratio = 2, a_factory=None, q_factory=None, i_objective=None):
        """
        Initializer.

        Args:
            epsilon (float): Target precision for parameter a.
            alpha (float): Confidence level.
            method (string): Statistical method for confidence interval estimation.
            min_ratio (float): Minimal ratio q for FindNextK
            a_factory (CircuitFactory): A oracle.
            q_factory (CircuitFactory): Q oracle.
            i_objective (int): Index of objective qubit.
        """
        self.validate(locals())
        super().__init__(a_factory, q_factory, i_objective)

        # store parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._method = method
        self._min_ratio = min_ratio

        # intialize internal parameters
        self._circuits = []
        self._ret = {}

        #intialize temporary variables and apriori parameter estimates
        self._i = 0
        self._k = [0]  # initial power of Q^k
        self._q = [] #multiplication factor
        self._theta_interval = [[0,1/4]] #apriori knowledge of theta/2/pi
        self._theta_i_interval = [[0,1/4]] #apriori knowledge of theta_i/2/pi
        self._a_interval = [[0,1]] #apriori knowledge of a parameter
        self._a_i_interval = [[0,1]] #apriori knowledge of a_i parameter
        self._up = [True] #initially theta is in the upper half-circle
        self._T = int(np.log(self._min_ratio*np.pi/8/self._epsilon)/np.log(self._min_ratio))+1
        self._num_oracle_queries = 0
        self._N_1_shots = [] #track number of 1 shots in each iteration

    @property
    def precision(self):
        return self._epsilon

    @precision.setter
    def precision(self, epsilon):
        self._epsilon = epsilon

    def find_next_k(self, k, up, theta_interval, min_ratio = 2):
        """
        Find the largest integer k, such that the scaled interval (4k + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k (int): Current power of the Q operator.
            up (bool): Boolean flag of wheather theta lies in upper half-circle or not.
            theta_interval (tuple(float, float)): Current confidence interval for the angle
                theta, i.e. (theta_lower, theta_upper)
            min_ratio (float): Minimal ratio K/K_i allowed in the algorithm

        Returns:
            tuple(int, bool): Next power k, and boolean flag for extrapoled interval

        """
        if min_ratio <= 1:
            raise AquaError('min_ratio must be larger than 1: '
                            'the next k should not be smaller than the previous one')

        # intialize variables
        theta_u = theta_interval[1]
        theta_l = theta_interval[0]
        K_current = 4*k+2

        # feasible K is not bigger than K_max, which is bounded by the length of current CI
        K_max = int(1/(2*(theta_u-theta_l)))
        K = K_max - (K_max-2)%4

        # find next feasible K = 4k+2
        while K >= self._min_ratio*K_current:
            theta_min = K*theta_l - int(K*theta_l)
            theta_max = K*theta_u - int(K*theta_u)
            if  theta_max <= 1/2 and theta_min <= 1/2 and theta_max >= theta_min:
                #if extrapolated theta in upper half-circle
                up = True
                return ((K-2)/4, up)
            elif theta_max >= 1/2 and theta_min >= 1/2 and theta_max >= theta_min:
                #if extrapolated theta in lower half-circle
                up = False
                return ((K-2)/4, up)
            K = K - 4
        #if algorithm does not find new feasible k, return old k
        return (k, up)

    def construct_circuit(self, k, measurement=False):
        """
        Construct the circuit Q^k A |0>

        Args:
            k (int): the power of Q
            measurement (bool): boolean flag to indicate if measurements should be included in the
                circuits

        Returns:
            QuantumCircuit: the circuit Q^k A |0>
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

    def _prob_from_cos_data_qasm(self, counts):
        #sometimes we have zero counts for some label, we need to fix this
        if len(list(counts.keys())) == 1:
            if list(counts.keys())[0] == '0':
                counts['1'] = 0
            else:
                counts['0'] = 0
        #estimate the probability
        prob = counts['1']/sum(counts.values())
        return prob

    def _prob_from_cos_data_sv(self, state_vector):
        prob = 0
        for i, g in enumerate(state_vector):
            if ("{:0%db}"%self._num_qubits).format(i)[-(1+self.i_objective)] == '1':
                prob = prob + np.abs(g)**2
        return prob

    def _chernoff(self, a, N_total_shots, T, alpha):
        epsilon_a = np.sqrt(3*np.log(2*T/alpha)/N_total_shots)
        if a - epsilon_a < 0:
            a_min = 0
        else:
            a_min = a - epsilon_a
        if a + epsilon_a > 1:
            a_max = 1
        else:
            a_max = a + epsilon_a
        return (a_min, a_max)

    def _run(self):
        self.check_factories()

        #initialize result variable for keeping the data from measurements
        if self._i==0:
            if self._quantum_instance.is_statevector:
                self._ret['statevectors'] = []
                self._sim_ind = 'sv'
            else:
                self._ret['counts'] = []
                #qasm -> counts
                self._sim_ind = 'qasm'
        else:
            if self._quantum_instance.is_statevector and self._sim_ind=='qasm':
                raise AquaError("QASM simlator was used before this run. Change qunatum_instance to QASM simulator")
            if not self._quantum_instance.is_statevector and self._sim_ind=='sv':
                raise AquaError("Statevector simlator was used before this run. Change qunatum_instance to statevector simulator")


        #do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
        while self._theta_interval[-1][1] - self._theta_interval[-1][0] > self._epsilon/np.pi:
            self._i += 1

            k, up = self.find_next_k(self._k[-1], self._up[-1], self._theta_interval[-1], min_ratio = self._min_ratio)
            self._k.append(k)
            self._up.append(up)
            self._q.append((2*self._k[-1]+1)/(2*self._k[-2]+1))

            #run measurements for Q^k A|0> circuit
            if self._quantum_instance.is_statevector:
                # run circuit on statevector simlator
                self._circuits.append(self.construct_circuit(k,measurement=False))
                ret = self._quantum_instance.execute(self._circuits[-1])

                # get statevector
                state_vector = np.asarray(ret.get_statevector(self._circuits[-1]))
                self._ret['statevectors'] += [{self._i : state_vector}]

                # calculate the probability of measuring '1'
                prob = self._prob_from_cos_data_sv(state_vector)
                # we imitate that we have 100 shots, calculate CI accordingly
                self._N_1_shots.append(int(prob*N_shots))

            else:
                # run circuit on QASM simulator
                self._circuits.append(self.construct_circuit(k,measurement=True))
                ret = self._quantum_instance.execute(self._circuits[-1])

                result_mmt = ret.get_counts(self._circuits[-1])
                self._ret['counts'] += [{self._i : result_mmt}]

                # calculate the probability of measuring '1'
                prob = self._prob_from_cos_data_qasm(result_mmt)
                self._N_1_shots.append(result_mmt['1'])


            # track number of Q-oracle calls
            N_shots = self._quantum_instance._run_config.shots
            self._num_oracle_queries += N_shots * k

            #if on previous iterations we have K_{i-1}==K_i, then we need to sum up these samples
            l=1
            if self._i>1:
                while self._k[self._i-l]==self._k[self._i] and self._i>=l+1:
                    l=l+1
                N_total_1_shots = sum([self._N_1_shots[j] for j in range(self._i-l,self._i)])
                N_total_shots = l*N_shots
            else:
                N_total_1_shots = self._N_1_shots[-1]
                N_total_shots = N_shots

            # compute a_min_i, a_max_i
            if self._method == 'chernoff':
                a_i = N_total_1_shots/N_total_shots
                a_i_min, a_i_max = self._chernoff(a_i, N_total_shots, self._T, self._alpha)
                self._a_i_interval.append([a_i_min,a_i_max])
            else:
                a_i_min, a_i_max = proportion_confint(N_total_1_shots, N_total_shots, method=self._method,alpha=self._alpha/self._T)
                self._a_i_interval.append([a_i_min,a_i_max])

            # compute theta_min_i, theta_max_i
            if up:
                theta_min_i = np.arccos(1-2*a_i_min)/2/np.pi
                theta_max_i = np.arccos(1-2*a_i_max)/2/np.pi
            else:
                theta_min_i = 1-np.arccos(1-2*a_i_max)/2/np.pi
                theta_max_i = 1-np.arccos(1-2*a_i_min)/2/np.pi

            self._theta_i_interval.append([theta_min_i,theta_max_i])

            # compute theta_u_i, theta_l_i
            K_i = 4*k+2 #current K_i factor

            theta_u_i = (int(K_i*self._theta_interval[-1][1])+theta_max_i)/K_i
            theta_l_i = (int(K_i*self._theta_interval[-1][0])+theta_min_i)/K_i

            self._theta_interval.append([theta_l_i, theta_u_i])

            # compute a_u_i, a_l_i
            a_u_i = np.sin(2*np.pi*theta_u_i)**2
            a_l_i = np.sin(2*np.pi*theta_l_i)**2

            self._a_interval.append([a_l_i, a_u_i])

        # set up results dictionary
        results = {
            'a_interval': self._a_interval,
            'theta_interval': self._theta_interval,
            'a_l': self._a_interval[-1][0],
            'a_u': self._a_interval[-1][1],
            'epsilon': (self._a_interval[-1][1]-self._a_interval[-1][0])/2,
            'theta_u': self._theta_interval[-1][1],
            'theta_l': self._theta_interval[-1][0],
            'k': self._k,
            'q': self._q,
            'num_oracle_queries': self._num_oracle_queries
        }

        return results

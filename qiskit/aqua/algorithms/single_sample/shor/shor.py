# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
The Shor's Factoring algorithm.
"""

import math
import array
import fractions
import logging
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit.aqua.utils.arithmetic import is_power
from qiskit.aqua import AquaError, Pluggable
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.circuits import FourierTransformCircuits as ftc


logger = logging.getLogger(__name__)


class Shor(QuantumAlgorithm):
    """
    The Shor's Factoring algorithm.

    Adapted from https://github.com/ttlion/ShorAlgQiskit
    """

    PROP_N = 'N'

    CONFIGURATION = {
        'name': 'Shor',
        'description': "The Shor's Factoring Algorithm",
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'shor_schema',
            'type': 'object',
            'properties': {
                PROP_N: {
                    'type': 'integer',
                    'default': 15,
                    'minimum': 3
                },
            },
            'additionalProperties': False
        },
        'problems': ['factoring'],
    }

    def __init__(self, N=15):
        """
        Constructor.

        Args:
            N (int): The integer to be factored.
        """
        self.validate(locals())
        super().__init__()

        # check the input integer
        if N < 1 or N % 2 == 0:
            raise AquaError('The input needs to be an odd integer greater than 1.')

        # check if the input integer is a power
        tf, b, p = is_power(N, return_decomposition=True)
        if tf:
            raise NotImplementedError

        self._N = N
        self._ret = {'factors': []}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params: parameters dictionary
            algo_input: input instance
        """

        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        shor_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        N = shor_params.get(Shor.PROP_N)

        return cls(N)

    def _get_angles(self, a):
        """
        Calculate the array of angles to be used in the addition in Fourier Space
        """
        s = bin(int(a))[2:].zfill(self._n + 1)
        angles = np.zeros([self._n + 1])
        for i in range(0, self._n + 1):
            for j in range(i, self._n + 1):
                if s[j] == '1':
                    angles[self._n - i] += math.pow(2, -(j - i))
            angles[self._n - i] *= np.pi
        return angles

    def _phi_add(self, circuit, q, inverse=False):
        """
        Creation of the circuit that performs addition by a in Fourier Space
        Can also be used for subtraction by setting the parameter inverse=True
        """
        angle = self._get_angles(self._N)
        for i in range(0, self._n + 1):
            circuit.u1(-angle[i] if inverse else angle[i], q[i])

    def _controlled_phi_add(self, circuit, q, ctl, inverse=False):
        """
        Single controlled version of the _phi_add circuit
        """
        angles = self._get_angles(self._N)
        for i in range(0, self._n + 1):
            angle = (-angles[i] if inverse else angles[i]) / 2

            circuit.u1(angle, ctl)
            circuit.cx(ctl, q[i])
            circuit.u1(-angle, q[i])
            circuit.cx(ctl, q[i])
            circuit.u1(angle, q[i])

    def _controlled_controlled_phi_add(self, circuit, q, ctl1, ctl2, a, inverse=False):
        """
        Doubly controlled version of the _phi_add circuit
        """

        # TODO: extract this gate
        def ccphase(circuit, angle, ctl1, ctl2, tgt):
            """
            Creation of a doubly controlled phase gate
            """

            angle /= 4

            circuit.u1(angle, ctl1)
            circuit.cx(ctl1, tgt)
            circuit.u1(-angle, tgt)
            circuit.cx(ctl1, tgt)
            circuit.u1(angle, tgt)

            circuit.cx(ctl2, ctl1)

            circuit.u1(-angle, ctl1)
            circuit.cx(ctl1, tgt)
            circuit.u1(angle, tgt)
            circuit.cx(ctl1, tgt)
            circuit.u1(-angle, tgt)

            circuit.cx(ctl2, ctl1)

            circuit.u1(angle, ctl2)
            circuit.cx(ctl2, tgt)
            circuit.u1(-angle, tgt)
            circuit.cx(ctl2, tgt)
            circuit.u1(angle, tgt)

        angle = self._get_angles(a)
        for i in range(self._n + 1):
            ccphase(circuit, -angle[i] if inverse else angle[i], ctl1, ctl2, q[i])

    def _controlled_controlled_phi_add_mod_N(self, circuit, q, ctl1, ctl2, aux, a):
        """
        Circuit that implements doubly controlled modular addition by a
        """
        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a)
        self._phi_add(circuit, q, inverse=True)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )
        circuit.cx(q[self._n], aux)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )
        self._controlled_phi_add(circuit, q, aux)

        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a, inverse=True)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )
        circuit.u3(np.pi, 0, np.pi, q[self._n])
        circuit.cx(q[self._n], aux)
        circuit.u3(np.pi, 0, np.pi, q[self._n])
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )
        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a)

    def _controlled_controlled_phi_add_mod_N_inv(self, circuit, q, ctl1, ctl2, aux, a):
        """
        Circuit that implements the inverse of doubly controlled modular addition by a
        """
        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a, inverse=True)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )
        circuit.u3(np.pi, 0, np.pi, q[self._n])
        circuit.cx(q[self._n], aux)
        circuit.u3(np.pi, 0, np.pi, q[self._n])
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )
        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a)
        self._controlled_phi_add(circuit, q, aux, inverse=True)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )
        circuit.cx(q[self._n], aux)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[q[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )
        self._phi_add(circuit, q)
        self._controlled_controlled_phi_add(circuit, q, ctl1, ctl2, a, inverse=True)

    def _controlled_multiple_mod_N(self, circuit, ctl, q, aux, a):
        """
        Circuit that implements single controlled modular multiplication by a
        """
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[aux[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )

        for i in range(0, self._n):
            self._controlled_controlled_phi_add_mod_N(
                circuit,
                aux,
                q[i],
                ctl,
                aux[self._n + 1],
                (2 ** i) * a % self._N
            )
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[aux[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )

        for i in range(0, self._n):
            circuit.cswap(ctl, q[i], aux[i])

        def modinv(a, m):
            def egcd(a, b):
                if a == 0:
                    return (b, 0, 1)
                else:
                    g, y, x = egcd(b % a, a)
                    return (g, x - (b // a) * y, y)

            g, x, y = egcd(a, m)
            if g != 1:
                raise Exception('modular inverse does not exist')
            else:
                return x % m

        a_inv = modinv(a, self._N)
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[aux[i] for i in reversed(range(self._n + 1))],
            do_swaps=False
        )

        for i in reversed(range(self._n)):
            self._controlled_controlled_phi_add_mod_N_inv(
                circuit,
                aux,
                q[i],
                ctl,
                aux[self._n + 1],
                math.pow(2, i) * a_inv % self._N
            )
        ftc.construct_circuit(
            circuit=circuit,
            qubits=[aux[i] for i in reversed(range(self._n + 1))],
            do_swaps=False,
            inverse=True
        )

    def construct_circuit(self):
        """Construct circuit.

        Returns:
            QuantumCircuit: quantum circuit.
        """

        # Get n value used in Shor's algorithm, to know how many qubits are used
        self._n = math.ceil(math.log(self._N, 2))

        logger.info(f'Total number of qubits: {4 * self._n + 2}.')

        # quantum register where the sequential QFT is performed
        self._up_qreg = QuantumRegister(2 * self._n, name='up')
        # quantum register where the multiplications are made
        self._down_qreg = QuantumRegister(self._n, name='down')
        # auxilliary quantum register used in addition and multiplication
        self._aux_qreg = QuantumRegister(self._n + 2, name='aux')

        # Create Quantum Circuit
        circuit = QuantumCircuit(self._up_qreg, self._down_qreg, self._aux_qreg)

        # Initialize down register to 1 and create maximal superposition in top register
        circuit.u2(0, np.pi, self._up_qreg)
        circuit.u3(np.pi, 0, np.pi, self._down_qreg[0])

        # Apply the multiplication gates as showed in the report in order to create the exponentiation
        for i in range(0, 2 * self._n):
            self._controlled_multiple_mod_N(
                circuit,
                self._up_qreg[i],
                self._down_qreg,
                self._aux_qreg,
                int(pow(self._a, pow(2, i)))
            )

        # Apply inverse QFT
        ftc.construct_circuit(circuit=circuit, qubits=self._up_qreg, do_swaps=True, inverse=True)

        return circuit

    def _pick_coprime_a(self):
        """
        Pick N's coprime number a (1 < a < N)
        """

        # Start with a=2
        a = 2

        # Get the smallest a such that a and N are coprime
        while math.gcd(a, self._N) != 1:
            a = a + 1

        return a

    def _get_factors(self, x_value, t_upper):
        """
        Apply the continued fractions to find r and the gcd to find the desired factors.
        """

        if x_value <= 0:
            logger.debug('x_value is <= 0, there are no continued fractions\n')
            return False

        logger.debug('Running continued fractions for this case\n')

        # Calculate T and x/T
        T = pow(2, t_upper)

        x_over_T = x_value / T

        # Cycle in which each iteration corresponds to putting one more term in the
        # calculation of the Continued Fraction (CF) of x/T

        # Initialize the first values according to CF rule
        i = 0
        b = array.array('i')
        t = array.array('f')

        b.append(math.floor(x_over_T))
        t.append(x_over_T - b[i])

        while i >= 0:

            # From the 2nd iteration onwards, calculate the new terms of the CF based
            # on the previous terms as the rule suggests

            if i > 0:
                b.append(math.floor(1 / t[i - 1]))
                t.append((1 / t[i - 1]) - b[i])

            # Calculate the CF using the known terms

            aux = 0
            j = i
            while j > 0:
                aux = 1 / (b[j] + aux)
                j = j - 1

            aux = aux + b[0]

            # Get the denominator from the value obtained
            frac = fractions.Fraction(aux).limit_denominator()
            den = frac.denominator

            logger.debug('Approximation number {0} of continued fractions:'.format(i + 1))
            logger.debug("Numerator:{0} \t\t Denominator: {1}\n".format(frac.numerator, frac.denominator))

            # Increment i for next iteration
            i = i + 1

            if den % 2 == 1:
                if i >= self._N:
                    logger.debug('Returning because have already done too much tries')
                    return False
                logger.debug('Odd denominator, will try next iteration of continued fractions\n')
                continue

            # If denominator even, try to get factors of N

            # Get the exponential a^(r/2)

            exponential = 0

            if den < 1000:
                exponential = pow(self._a, den / 2)

            # Check if the value is too big or not
            if math.isinf(exponential) or exponential > 1000000000:
                logger.debug('Denominator of continued fraction is too big!\n')
                aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
                if aux_out != '1':
                    return False
                else:
                    continue

            # If the value is not to big (infinity), then get the right values and do the proper gcd()

            putting_plus = int(exponential + 1)

            putting_minus = int(exponential - 1)

            one_factor = math.gcd(putting_plus, self._N)
            other_factor = math.gcd(putting_minus, self._N)

            # Check if the factors found are trivial factors or are the desired factors

            if one_factor == 1 or one_factor == self._N or other_factor == 1 or other_factor == self._N:
                logger.debug('Found just trivial factors, not good enough\n')
                # Check if the number has already been found, use i-1 because i was already incremented
                if t[i - 1] == 0:
                    logger.debug('The continued fractions found exactly x_final/(2^(2n)) , leaving funtion\n')
                    return False
                if i < self._N:
                    aux_out = input('Input number 1 if you want to continue searching, other if you do not: ')
                    if aux_out != '1':
                        return False
                else:
                    # Return if already too much tries and numbers are huge
                    logger.debug('Returning because have already done too many tries\n')
                    return False
            else:
                logger.debug('The factors of {0} are {1} and {2}\n'.format(self._N, one_factor, other_factor))
                logger.debug('Found the desired factors!\n')
                factors = sorted((one_factor, other_factor))
                if factors not in self._ret['factors']:
                    self._ret['factors'].append(factors)
                return True

    def _run(self):
        self._a = self._pick_coprime_a()
        logger.debug('Running with N={} and a={}\n'.format(self._N, self._a))

        circuit = self.construct_circuit()

        if self._quantum_instance.is_statevector:
            logger.warning('The statevector_simulator might lead to subsequent computation using too much memory.')
            result = self._quantum_instance.execute(circuit)
            complete_state_vec = result.get_statevector(circuit)
            # TODO: this uses too much memory
            up_qreg_density_mat = get_subsystem_density_matrix(
                complete_state_vec,
                range(2 * self._n, 4 * self._n + 2)
            )
            up_qreg_density_mat_diag = np.diag(up_qreg_density_mat)

            counts = dict()
            for i, v in enumerate(up_qreg_density_mat_diag):
                if not v == 0:
                    counts[bin(int(i))[2:].zfill(2 * self._n)] = v ** 2
        else:
            up_cqreg = ClassicalRegister(2 * self._n, name='m')
            circuit.add_register(up_cqreg)
            circuit.measure(self._up_qreg, up_cqreg)
            counts = self._quantum_instance.execute(circuit).get_counts(circuit)

        # For each simulation result, print proper info to user and try to calculate the factors of N
        for output_desired in list(counts.keys()):
            # Get the x_value from the final state qubits
            logger.info("------> Analyzing result {0}.".format(output_desired))
            x_value = int(output_desired, 2)
            logger.info('In decimal, x_final value for this result is: {0}\n'.format(x_value))
            success = self._get_factors(int(x_value), int(2 * self._n))
            logger.info('success: ', success)

        return self._ret

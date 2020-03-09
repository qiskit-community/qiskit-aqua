# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.iqfts import Standard as IQFT
import numpy as np


class OptimizationProblemToNegativeValueOracle:

    def __init__(self, num_output_qubits, verbose=False, backend='statevector_simulator'):
        """ A helper function that converts a quadratic function into a state preparation operator A (InitialState) and
            oracle O (CustomCircuitOracle) that recognizes negative numbers.
            :param num_output_qubits: The number of qubits required to represent the output.
            :param verbose: Verbose flag.
            :param backend: A string corresponding to a valid Qiskit backend.
        """
        self._num_value = num_output_qubits
        self._verbose = verbose
        self._backend = backend
        self._largest = 1  # If self.approximate, this value is used to approximate to and from non-integers.

    def encode(self, linear_coeff, quadratic_coeff, constant):
        """ Takes the coefficients and constants of a quadratic function and returns a state preparation operator A
            (InitialState.Custom) that encodes the function, and an oracle O (Oracle.CustomCircuitOracle) that
            recognizes negative values.
            :param linear_coeff: (n x 1 matrix) The linear coefficients of the function.
            :param quadratic_coeff: (n x n matrix) The quadratic coefficients of the function.
            :param constant: (int) The constant of the function.
        """
        # Get circuit requirements from input.
        num_key = len(linear_coeff)
        num_ancilla = max(num_key, self._num_value) - 1

        # Get the function dictionary.
        f = self.__get_function(num_key, linear_coeff, quadratic_coeff, constant)
        if self._verbose:
            print("f:", f, "\n")

        # Define state preparation operator A from function.
        qd = QQUBODictionary(num_key, self._num_value, num_ancilla, f, backend=self._backend)
        a_operator_circuit = qd.circuit
        a_operator = Custom(a_operator_circuit.width(), circuit=a_operator_circuit)

        # Get registers from the A operator circuit.
        reg_map = {}
        for reg in a_operator_circuit.qregs:
            reg_map[reg.name] = reg
        key_val = reg_map["key_value"]
        anc = reg_map["ancilla"]

        # Build negative value oracle O.
        oracle_bit = QuantumRegister(1, "oracle")
        oracle_circuit = QuantumCircuit(key_val, anc, oracle_bit)
        self.__cxzxz(oracle_circuit, key_val[num_key], oracle_bit[0])
        oracle = CustomCircuitOracle(variable_register=key_val,
                                     output_register=oracle_bit,
                                     ancillary_register=anc,
                                     circuit=oracle_circuit,
                                     evaluate_classically_callback=self.__evaluate_classically)

        return a_operator, oracle, f

    @staticmethod
    def __get_function(num_assets, linear, quadratic, constant):
        """Convert the problem to a dictionary format."""
        f = {-1: constant}
        for i in range(num_assets):
            i_ = i
            f[i_] = -linear[i_]
            for j in range(i):
                j_ = j
                f[(i_, j_)] = int(quadratic[i_, j_])

        return f

    @staticmethod
    def __cxzxz(circuit, ctrl, tgt):
        """Multiplies by -1."""
        circuit.cx(ctrl, tgt)
        circuit.cz(ctrl, tgt)
        circuit.cx(ctrl, tgt)
        circuit.cz(ctrl, tgt)

    def __evaluate_classically(self, measurement):
        """ evaluate classical """
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1],
                                                                       range(len(measurement)))]
        assignment_dict = dict()
        for v in assignment:
            assignment_dict[v] = bool(v < 0)
        return assignment_dict, assignment

class QDictionary:
    """
        A parent class that defines a Quantum Dictionary, which encodes key-value pairs into the quantum state. See
        https://arxiv.org/abs/1912.04088 for a formal definition.
    """

    def __init__(self, key_bits, value_bits, ancilla_bits, f, prepare, backend="statevector_simulator"):
        """
            Initializes a Quantum Dictionary.
            :param key_bits: The number of key bits.
            :param value_bits: The number of value bits.
            :param ancilla_bits: The number of precision (result) bits.
            :param f: The function to encode, can be a list or a lambda.
            :param prepare: A method that encodes f into the quantum state.
        """
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.ancilla_bits = ancilla_bits
        self.f = f
        self.prepare = prepare
        self.backend = backend

    def construct_circuit(self):
        """
            Creates a circuit for the initialized Quantum Dictionary.
            :return: A QuantumCircuit object describing the Quantum Dictionary.
        """
        key_val = QuantumRegister(self.key_bits + self.value_bits, "key_value")
        ancilla = QuantumRegister(self.ancilla_bits, "ancilla")

        if self.backend == "statevector_simulator":
            circuit = QuantumCircuit(key_val, ancilla)
        else:
            measure = ClassicalRegister(self.key_bits + self.value_bits)
            circuit = QuantumCircuit(key_val, ancilla, measure)

        self.prepare(self.f, circuit, key_val)

        return circuit


class QQUBODictionary(QDictionary):
    """
        A QDictionary that creates a state preparation operator for a given QUBO problem.
    """

    def __init__(self, key_bits, value_bits, ancilla_bits, f, backend="statevector_simulator"):
        QDictionary.__init__(self, key_bits, value_bits, ancilla_bits, f, self.prepare_quadratic, backend=backend)
        self._circuit = None

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = self.construct_circuit()
        return self._circuit

    def prepare_quadratic(self, d, circuit, key_val):
        """ Encodes a QUBO in the proper dictionary format into the state of a given register.
        :param d: (dict) The QUBO problem. The keys should be the subscripts of the coefficients (e.g. x_1 -> 1), with
            the constant (if present) being represented with a key of -1 (i.e. d[-1] = constant). Quadratic coefficients
            should use a tuple for the key, with the corresponding subscripts inside (e.g. 2*x_1*x_2 -> d[(1,2)]=2).
        :param circuit: The QuantumCircuit to apply the operator to.
        :param key_val: The QuantumRegister containing the key and value qubits. They are combined here to follow the
            register format expected by algorithms like Qiskit Aqua's Grover.
        """
        circuit.h(key_val)

        # Linear Coefficients
        for i in range(self.value_bits):
            if d.get(-1, 0) != 0:
                circuit.u1(1/2 ** self.value_bits * 2 * np.pi * 2 ** i * d[-1], key_val[self.key_bits + i])
            for j in range(self.key_bits):
                if d.get(j, 0) != 0:
                    circuit.cu1(1/2 ** self.value_bits * 2 * np.pi * 2 ** i * d[j], key_val[j],
                                key_val[self.key_bits + i])

        # Quadratic Coefficients
        for i in range(self.value_bits):
            for k, v in d.items():
                if isinstance(k, tuple):
                    circuit.mcu1(1/2 ** self.value_bits * 2 * np.pi * 2 ** i * v, [key_val[k[0]], key_val[k[1]]],
                                 key_val[self.key_bits + i])

        iqft = IQFT(self.value_bits)
        value = [key_val[v] for v in range(self.key_bits, self.key_bits + self.value_bits)]
        iqft.construct_circuit(qubits=value, circuit=circuit)

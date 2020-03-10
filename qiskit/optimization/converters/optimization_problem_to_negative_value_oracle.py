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

"""OptimizationProblemToNegativeValueOracle module"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.iqfts import Standard as IQFT


class OptimizationProblemToNegativeValueOracle:

    """
        Converts an optimization problem (QUBO) to a negative value oracle and state preparation
        operators.
    """

    def __init__(self, num_output_qubits, verbose=False, backend='statevector_simulator'):
        """
        Constructor.
        Args:
            num_output_qubits (int): The number of qubits required to represent the output.
            verbose (bool, optional): Verbose flag - prints/plots state at each iteration of GAS.
            backend (str, optional): A string corresponding to the name of the selected backend.
        """
        self._num_value = num_output_qubits
        self._verbose = verbose
        self._backend = backend

    def encode(self, linear_coeff, quadratic_coeff, constant):
        """
        A helper function that converts a quadratic function into a state preparation operator A and
         oracle O that recognizes negative numbers.
        Args:
            linear_coeff (np.array): The linear coefficients of the QUBO.
            quadratic_coeff (np.array): The quadratic coefficients of the QUBO.
            constant (int): The constant of the QUBO.
        Returns:
            Tuple(InitialState.Custom, CustomCircuitOracle, dict): A state preparation operator A
            (InitialState), an oracle O (CustomCircuitOracle) that recognizes negative numbers, and
            a dictionary representation of the function coefficients, where the key -1 represents
            the constant.
        """
        # Get circuit requirements from input.
        num_key = len(linear_coeff)
        num_ancilla = max(num_key, self._num_value) - 1

        # Get the function dictionary.
        func = self.__get_function(num_key, linear_coeff, quadratic_coeff, constant)
        if self._verbose:
            print("f:", func, "\n")

        # Define state preparation operator A from function.
        quantum_dict = QQUBODictionary(num_key, self._num_value, num_ancilla,
                                       func, backend=self._backend)
        a_operator_circuit = quantum_dict.circuit
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

        return a_operator, oracle, func

    @staticmethod
    def __get_function(num_assets, linear, quadratic, constant):
        """Convert the problem to a dictionary format."""
        func = {-1: constant}
        for i in range(num_assets):
            func[i] = -linear[i]
            for j in range(i):
                func[(i, j)] = int(quadratic[i, j])

        return func

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


class QuantumDictionary:
    """
        A parent class that defines a Quantum Dictionary, which encodes key-value pairs into the
        quantum state. See https://arxiv.org/abs/1912.04088 for a formal definition.
    """

    def __init__(self, key_bits, value_bits, ancilla_bits, func_dict, prepare,
                 backend="statevector_simulator"):
        """
        Constructor.
        Args:
            key_bits (int): The number of key bits.
            value_bits (int): The number of value bits.
            ancilla_bits (int): The number of precision (result) bits.
            func_dict (dict): The dictionary of function coefficients to encode.
            prepare (function): A method that encodes f into the quantum state.
            backend (str, optional): A string corresponding to the name of the selected backend.
        """
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.ancilla_bits = ancilla_bits
        self.func_dict = func_dict
        self.prepare = prepare
        self.backend = backend

    def construct_circuit(self):
        """
        Creates a circuit for the initialized Quantum Dictionary.
        Returns:
            QuantumCircuit: Circuit object describing the Quantum Dictionary.
        """
        key_val = QuantumRegister(self.key_bits + self.value_bits, "key_value")
        ancilla = QuantumRegister(self.ancilla_bits, "ancilla")

        if self.backend == "statevector_simulator":
            circuit = QuantumCircuit(key_val, ancilla)
        else:
            measure = ClassicalRegister(self.key_bits + self.value_bits)
            circuit = QuantumCircuit(key_val, ancilla, measure)

        self.prepare(self.func_dict, circuit, key_val)

        return circuit


class QQUBODictionary(QuantumDictionary):
    """
        A Quantum Dictionary that creates a state preparation operator for a given QUBO problem.
    """

    def __init__(self, key_bits, value_bits, ancilla_bits, func_dict,
                 backend="statevector_simulator"):
        """
        Constructor.
        Args:
            key_bits (int): The number of key bits.
            value_bits (int): The number of value bits.
            ancilla_bits (int): The number of precision (result) bits.
            func_dict (dict): The dictionary of function coefficients to encode.
            backend (str, optional): A string corresponding to the name of the selected backend.
        """
        QuantumDictionary.__init__(self, key_bits, value_bits, ancilla_bits, func_dict,
                                   self.prepare_quadratic, backend=backend)
        self._circuit = None

    @property
    def circuit(self):
        """
        Provides the circuit of the Quantum Dictionary. Will construct one if not yet created.
        Returns:
            QuantumCircuit: Circuit object describing the Quantum Dictionary.
        """
        if self._circuit is None:
            self._circuit = self.construct_circuit()
        return self._circuit

    def prepare_quadratic(self, func_dict, circuit, key_val):
        """
        Encodes a QUBO in the proper dictionary format into the state of a given register.
        Args:
            func_dict (dict): The QUBO problem. The keys should be subscripts of the coefficients
                (e.g. x_1 -> 1), with the constant (if present) being represented with a key of -1
                (i.e. d[-1] = constant). Quadratic coefficients should use a tuple for the key, with
                 the corresponding subscripts inside (e.g. 2*x_1*x_2 -> d[(1,2)]=2).
            circuit (QuantumCircuit): The circuit to apply the operator to.
            key_val (QuantumRegister): Register containing the key and value qubits. They are
                combined here to follow the register format expected by algorithms like Qiskit
                Aqua's Grover.
        """
        circuit.h(key_val)

        # Linear Coefficients
        for i in range(self.value_bits):
            if func_dict.get(-1, 0) != 0:
                circuit.u1(1 / 2 ** self.value_bits * 2 * np.pi * 2 ** i * func_dict[-1],
                           key_val[self.key_bits + i])
            for j in range(self.key_bits):
                if func_dict.get(j, 0) != 0:
                    circuit.cu1(1 / 2 ** self.value_bits * 2 * np.pi * 2 ** i * func_dict[j],
                                key_val[j], key_val[self.key_bits + i])

        # Quadratic Coefficients
        for i in range(self.value_bits):
            for k, v in func_dict.items():
                if isinstance(k, tuple):
                    circuit.mcu1(1/2 ** self.value_bits * 2 * np.pi * 2 ** i * v,
                                 [key_val[k[0]], key_val[k[1]]], key_val[self.key_bits + i])

        iqft = IQFT(self.value_bits)
        value = [key_val[v] for v in range(self.key_bits, self.key_bits + self.value_bits)]
        iqft.construct_circuit(qubits=value, circuit=circuit)

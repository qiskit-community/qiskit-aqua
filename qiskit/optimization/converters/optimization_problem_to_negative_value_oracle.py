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

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Union, Callable, Any
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.iqfts import Standard as IQFT
from qiskit.providers import BaseBackend
from qiskit.optimization.problems import OptimizationProblem


class OptimizationProblemToNegativeValueOracle:

    """Converts an optimization problem (QUBO) to a negative value oracle.

    In addition, a state preparation operator is generated from the coefficients and constant of a
    QUBO, which can be used to encode the function into a quantum state. In conjuction, this oracle
    and operator can be used to flag the negative values of a QUBO encoded in a quantum state.
    """

    def __init__(self, num_output_qubits: int, backend: Optional[BaseBackend] = None) -> None:
        """
        Args:
            num_output_qubits: The number of qubits required to represent the output.
            backend: Instance of selected backend.
        """
        self._num_value = num_output_qubits
        if backend is None:
            self._backend = Aer.get_backend('statevector_simulator')
        else:
            self._backend = backend
        self._logger = logging.getLogger(__name__)

    def encode(self, problem: OptimizationProblem) -> \
            Tuple[Custom, CustomCircuitOracle, Dict[Union[int, Tuple[int, int]], int]]:
        """ A helper function that converts a QUBO into an oracle that recognizes negative numbers.

        Args:
            linear_coeff: The linear coefficients of the QUBO.
            quadratic_coeff: The quadratic coefficients of the QUBO.
            constant: The constant of the QUBO.

        Returns:
            A state preparation operator A, an oracle O that recognizes negative numbers, and
            a dictionary representation of the function coefficients, where the key -1 represents
            the constant.
        """

        # get linear part of objective
        linear_dict = problem.objective.get_linear()
        linear_coeff = np.zeros(problem.variables.get_num())
        for i, v in linear_dict.items():
            linear_coeff[i] = v

        # get quadratic part of objective
        quadratic_dict = problem.objective.get_quadratic()
        quadratic_coeff = {}
        for i, jv in quadratic_dict.items():
            for j, v in jv.items():
                coeff = quadratic_coeff.get((j, i), 0)
                if i <= j:
                    quadratic_coeff[(i, j)] = v / 2 + coeff  # divide by 2 since problem considers xQx/2.
                else:
                    quadratic_coeff[(j, i)] = v / 2 + coeff

        constant = problem.objective.get_offset()

        # Get circuit requirements from input.
        num_key = len(linear_coeff)
        # num_ancilla = max(num_key, self._num_value) - 1  # TODO: ancillas are not used, are they?
        num_ancilla = 0

        # Get the function dictionary.
        func = self._get_function(linear_coeff, quadratic_coeff, constant)
        self._logger.info("Function: {}\n", func)

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
        # anc = reg_map["ancilla"]

        # TODO: Can we use LogicalExpressionOracle instead?
        # Build negative value oracle O.
        oracle_bit = QuantumRegister(1, "oracle")
        oracle_circuit = QuantumCircuit(key_val, oracle_bit)
        # oracle_circuit = QuantumCircuit(key_val, anc, oracle_bit)  # TODO
        self._cxzxz(oracle_circuit, key_val[num_key], oracle_bit[0])
        oracle = CustomCircuitOracle(variable_register=key_val,
                                     output_register=oracle_bit,
                                    #  ancillary_register=anc,  # TODO
                                     circuit=oracle_circuit,
                                     evaluate_classically_callback=self._evaluate_classically)

        return a_operator, oracle, func

    @staticmethod
    def _get_function(linear: np.array, quadratic: np.array, constant: int) -> \
            Dict[Union[int, Tuple[int, int]], int]:
        """Convert the problem to a dictionary format."""
        func = {-1: constant}
        for i, v in enumerate(linear):
            func[i] = v
        for ij, v in quadratic.items():
            i, j = ij
            func[(i, j)] = int(quadratic[(i, j)])

        return func

    @staticmethod
    def _cxzxz(circuit: QuantumCircuit, ctrl: QuantumRegister, tgt: QuantumRegister) -> None:
        """Multiplies by -1."""
        circuit.cx(ctrl, tgt)
        circuit.cz(ctrl, tgt)
        circuit.cx(ctrl, tgt)
        circuit.cz(ctrl, tgt)

    def _evaluate_classically(self, measurement):
        # TODO: Typing for this method? Still not sure what it's used for. Required by Grover.
        """ evaluate classical """
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1],
                                                                       range(len(measurement)))]
        assignment_dict = dict()
        for v in assignment:
            assignment_dict[v] = bool(v < 0)
        return assignment_dict, assignment


class QuantumDictionary:

    """Defines a Quantum Dictionary, which encodes key-value pairs into the quantum state.

    See https://arxiv.org/abs/1912.04088 for a formal definition.
    """

    def __init__(self, key_bits: int, value_bits: int, ancilla_bits: int,
                 func_dict: Dict[Union[int, Tuple[int, int]], int],
                 prepare: Callable[[Any, QuantumCircuit, QuantumRegister], None],
                 backend: Optional[BaseBackend] = None) -> None:
        """
        Args:
            key_bits: The number of key bits.
            value_bits: The number of value bits.
            ancilla_bits: The number of precision (result) bits.
            func_dict: The dictionary of function coefficients to encode.
            prepare: A method that encodes f into the quantum state.
            backend: Instance of selected backend.
        """
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._ancilla_bits = ancilla_bits
        self._func_dict = func_dict
        self._prepare = prepare
        if backend is None:
            self._backend = Aer.get_backend('statevector_simulator')
        else:
            self._backend = backend

    def construct_circuit(self) -> QuantumCircuit:
        """Creates a circuit for the initialized Quantum Dictionary.

        Returns:
            Circuit object describing the Quantum Dictionary.
        """
        key_val = QuantumRegister(self._key_bits + self._value_bits, "key_value")
        # ancilla = QuantumRegister(self._ancilla_bits, "ancilla")  # TODO

        if self._backend.name == "statevector_simulator":
            # circuit = QuantumCircuit(key_val, ancilla)  # TODO
            circuit = QuantumCircuit(key_val)
        else:
            measure = ClassicalRegister(self._key_bits + self._value_bits)
            # circuit = QuantumCircuit(key_val, ancilla, measure)  # TODO
            circuit = QuantumCircuit(key_val, measure)

        self._prepare(self._func_dict, circuit, key_val)

        return circuit


class QQUBODictionary(QuantumDictionary):

    """ A Quantum Dictionary that creates a state preparation operator for a given QUBO problem.
    """

    def __init__(self, key_bits: int, value_bits: int, ancilla_bits: int,
                 func_dict: Dict[Union[int, Tuple[int, int]], int],
                 backend: Optional[BaseBackend] = None) -> None:
        """
        Args:
            key_bits: The number of key bits.
            value_bits: The number of value bits.
            ancilla_bits: The number of ancilla bits.
            func_dict: The dictionary of function coefficients to encode.
            backend: Instance of selected backend.
        """
        if backend is None:
            self._backend = Aer.get_backend('statevector_simulator')
        else:
            self._backend = backend
        QuantumDictionary.__init__(self, key_bits, value_bits, ancilla_bits, func_dict,
                                   self.prepare_quadratic, backend=backend)
        self._circuit = None

    @property
    def circuit(self) -> QuantumCircuit:
        """ Provides the circuit of the Quantum Dictionary. Will construct one if not yet created.

        Returns:
            Circuit object describing the Quantum Dictionary.
        """
        if self._circuit is None:
            self._circuit = self.construct_circuit()
        return self._circuit

    def prepare_quadratic(self, func_dict: Dict[Union[int, Tuple[int, int]], int],
                          circuit: QuantumCircuit, key_val: QuantumRegister) -> None:
        """Encodes a QUBO in the proper dictionary format into the state of a given register.

        Args:
            func_dict: Representation of the QUBO problem. The keys should be subscripts of the
                coefficients (e.g. x_1 -> 1), with the constant (if present) being represented with
                a key of -1 (i.e. d[-1] = constant). Quadratic coefficients should use a tuple for
                the key, with the corresponding subscripts inside (e.g. 2*x_1*x_2 -> d[(1,2)]=2).
            circuit: The circuit to apply the operator to.
            key_val: Register containing the key and value qubits. They are combined here to follow
            the register format expected by algorithms like Qiskit Aqua's Grover.
        """
        circuit.h(key_val)

        # Linear Coefficients
        for i in range(self._value_bits):
            if func_dict.get(-1, 0) != 0:
                circuit.u1(1 / 2 ** self._value_bits * 2 * np.pi * 2 ** i * func_dict[-1],
                           key_val[self._key_bits + i])
            for j in range(self._key_bits):
                if func_dict.get(j, 0) != 0:
                    circuit.cu1(1 / 2 ** self._value_bits * 2 * np.pi * 2 ** i * func_dict[j],
                                key_val[j], key_val[self._key_bits + i])

        # Quadratic Coefficients
        for i in range(self._value_bits):
            for k, v in func_dict.items():
                if isinstance(k, tuple):
                    circuit.mcu1(1 / 2 ** self._value_bits * 2 * np.pi * 2 ** i * v,
                                 [key_val[k[0]], key_val[k[1]]], key_val[self._key_bits + i])

        iqft = IQFT(self._value_bits)
        value = [key_val[v] for v in range(self._key_bits, self._key_bits + self._value_bits)]
        iqft.construct_circuit(qubits=value, circuit=circuit)

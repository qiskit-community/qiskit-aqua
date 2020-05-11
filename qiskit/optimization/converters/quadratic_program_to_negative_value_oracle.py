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

"""QuadraticProgramToNegativeValueOracle module"""

import logging
from typing import Tuple, Dict, Union

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.oracles import CustomCircuitOracle
from ..problems.quadratic_program import QuadraticProgram

logger = logging.getLogger(__name__)


class QuadraticProgramToNegativeValueOracle:
    """Converts an optimization problem (QUBO) to a negative value oracle.

    In addition, a state preparation operator is generated from the coefficients and constant of a
    QUBO, which can be used to encode the function into a quantum state. In conjunction, this oracle
    and operator can be used to flag the negative values of a QUBO encoded in a quantum state.

    The construction of the oracle is discussed in [1].

    References:
        [1]: Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
            arxiv:1912.04088.
    """

    def __init__(self, num_value_qubits: int, measurement: bool = False) -> None:
        """
        Args:
            num_value_qubits: The number of qubits required to represent the output.
            measurement: Whether the A operator contains measurements.
        """
        self._num_key = 0
        self._num_value = num_value_qubits
        self._measurement = measurement

    def encode(self, problem: QuadraticProgram) -> \
            Tuple[Custom, CustomCircuitOracle, Dict[Union[int, Tuple[int, int]], int]]:
        """A helper function that converts a QUBO into an oracle that recognizes negative numbers.

        Args:
            problem: The problem to be solved.

        Returns:
            A state preparation operator A, an oracle O that recognizes negative numbers, and
            a dictionary representation of the function coefficients, where the key -1 represents
            the constant.
        """

        # get linear part of objective
        linear_dict = problem.objective.linear.to_dict()
        linear_coeff = np.zeros(len(problem.variables))
        for i, v in linear_dict.items():
            linear_coeff[i] = v

        # get quadratic part of objective
        quadratic_coeff = problem.objective.quadratic.to_dict()

        constant = int(problem.objective.constant)

        # Get circuit requirements from input.
        self._num_key = len(linear_coeff)

        # Get the function dictionary.
        func = self._get_function(linear_coeff, quadratic_coeff, constant)
        logger.info("Function: %s\n", func)

        # Define state preparation operator A from function.
        a_operator_circuit = self._build_operator(func)
        a_operator = Custom(a_operator_circuit.width(), circuit=a_operator_circuit)

        # Get registers from the A operator circuit.
        reg_map = {}
        for reg in a_operator_circuit.qregs:
            reg_map[reg.name] = reg
        key_val = reg_map["key_value"]

        # Build negative value oracle O.
        oracle_bit = QuantumRegister(1, "oracle")
        oracle_circuit = QuantumCircuit(key_val, oracle_bit)
        oracle_circuit.z(key_val[self._num_key])  # recognize negative values.
        oracle = CustomCircuitOracle(variable_register=key_val,
                                     output_register=oracle_bit,
                                     circuit=oracle_circuit,
                                     evaluate_classically_callback=self._evaluate_classically)

        return a_operator, oracle, func

    @staticmethod
    def _get_function(linear: np.array, quadratic: np.array, constant: int) -> \
            Dict[Union[int, Tuple[int, int]], int]:
        """Convert the problem to a dictionary format."""
        func = {-1: int(constant)}
        for i, v in enumerate(linear):
            func[i] = int(v)
        for (i, j), v in quadratic.items():
            if i != j:
                func[(i, j)] = int(quadratic[(i, j)])
            else:
                func[i] += int(v)

        return func

    def _evaluate_classically(self, measurement):
        """ evaluate classical """
        value = measurement[self._num_key:self._num_key + self._num_value]
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement,
                                                                       range(len(measurement)))]
        evaluation = value[0] == '1'
        return evaluation, assignment

    def _build_operator(self, func_dict: Dict[Union[int, Tuple[int, int]], int]) -> QuantumCircuit:
        """Creates a circuit for the state preparation operator.

        Args:
            func_dict: Representation of the QUBO problem. The keys should be subscripts of the
                coefficients (e.g. x_1 -> 1), with the constant (if present) being represented with
                a key of -1 (i.e. d[-1] = constant). Quadratic coefficients should use a tuple for
                the key, with the corresponding subscripts inside (e.g. 2*x_1*x_2 -> d[(1,2)]=2).

        Returns:
            Circuit object describing the state preparation operator.
        """

        # Build initial circuit.
        key_val = QuantumRegister(self._num_key + self._num_value, "key_value")
        circuit = QuantumCircuit(key_val)
        if self._measurement:
            measure = ClassicalRegister(self._num_key + self._num_value)
            circuit.add_register(measure)
        circuit.h(key_val)

        # Linear Coefficients.
        for i in range(self._num_value):
            if func_dict.get(-1, 0) != 0:
                circuit.u1(1 / 2 ** self._num_value * 2 * np.pi * 2 ** i * func_dict[-1],
                           key_val[self._num_key + i])
            for j in range(self._num_key):
                if func_dict.get(j, 0) != 0:
                    circuit.cu1(1 / 2 ** self._num_value * 2 * np.pi * 2 ** i * func_dict[j],
                                key_val[j], key_val[self._num_key + i])

        # Quadratic Coefficients.
        for i in range(self._num_value):
            for k, v in func_dict.items():
                if isinstance(k, tuple):
                    a_v = [key_val[int(k[0])], key_val[int(k[1])]]
                    b_v = key_val[self._num_key + i]
                    circuit.mcu1(1 / 2 ** self._num_value * 2 * np.pi * 2 ** i * v,
                                 a_v, b_v)

        # Add IQFT. Adding swaps at the end of the IQFT, not the beginning.
        iqft = QFT(self._num_value, do_swaps=False).inverse()
        value = [key_val[v] for v in range(self._num_key, self._num_key + self._num_value)]
        circuit.compose(iqft, qubits=value, inplace=True)

        for i in range(len(value) // 2):
            circuit.swap(value[i], value[-(i + 1)])

        return circuit

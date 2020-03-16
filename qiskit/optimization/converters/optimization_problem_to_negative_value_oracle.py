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
from typing import Optional, Tuple, Dict, Union
import numpy as np
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
        self._num_key = 0
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
            problem: The problem to be solved.

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
                    # divide by 2 since problem considers xQx/2.
                    quadratic_coeff[(i, j)] = v / 2 + coeff
                else:
                    quadratic_coeff[(j, i)] = v / 2 + coeff

        constant = problem.objective.get_offset()

        # Get circuit requirements from input.
        self._num_key = len(linear_coeff)

        # Get the function dictionary.
        func = self._get_function(linear_coeff, quadratic_coeff, constant)
        self._logger.info("Function: {}\n", func)

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
        func = {-1: constant}
        for i, v in enumerate(linear):
            func[i] = v
        for ij, v in quadratic.items():
            i, j = ij
            if i != j:
                func[(i, j)] = int(quadratic[(i, j)])
            else:
                func[i] += v

        return func

    def _evaluate_classically(self, measurement):
        # TODO: Typing for this method? Still not sure what it's used for. Required by Grover.
        """ evaluate classical """
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1],
                                                                       range(len(measurement)))]
        assignment_dict = dict()
        for v in assignment:
            assignment_dict[v] = bool(v < 0)
        return assignment_dict, assignment

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
        if self._backend.name == "statevector_simulator":
            circuit = QuantumCircuit(key_val)
        else:
            measure = ClassicalRegister(self._num_key + self._num_value)
            circuit = QuantumCircuit(key_val, measure)
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
                    circuit.mcu1(1 / 2 ** self._num_value * 2 * np.pi * 2 ** i * v,
                                 [key_val[k[0]], key_val[k[1]]], key_val[self._num_key + i])

        # Add IQFT.
        iqft = IQFT(self._num_value)
        value = [key_val[v] for v in range(self._num_key, self._num_key + self._num_value)]
        iqft.construct_circuit(qubits=value, circuit=circuit)

        return circuit

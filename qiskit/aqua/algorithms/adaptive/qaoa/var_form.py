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

import numpy as np
from functools import reduce
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator, op_converter


class QAOAVarForm:
    """Global X phases and parameterized problem hamiltonian."""

    def __init__(self, cost_operator, p, initial_state=None, mixer_operator=None):
        """
        Constructor, following the QAOA paper https://arxiv.org/abs/1411.4028

        Args:
            cost_operator (WeightedPauliOperator): The operator representing the cost of the optimization problem,
                                                   denoted as U(B, gamma) in the original paper.
            p (int): The integer parameter p, which determines the depth of the circuit,
                     as specified in the original paper.
            initial_state (InitialState, optional): An optional initial state to use.
            mixer_operator (WeightedPauliOperator, optional): An optional custom mixer operator to use instead of
                                                              the global X-rotations, denoted as U(B, beta)
                                                              in the original paper.
        """
        cost_operator = op_converter.to_weighted_pauli_operator(cost_operator)
        self._cost_operator = cost_operator
        self._p = p
        self._initial_state = initial_state
        self.num_parameters = 2 * p
        self.parameter_bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p
        self.preferred_init_points = [0] * p * 2

        # prepare the mixer operator
        v = np.zeros(self._cost_operator.num_qubits)
        ws = np.eye(self._cost_operator.num_qubits)
        if mixer_operator is None:
            self._mixer_operator = reduce(
                lambda x, y: x + y,
                [
                    WeightedPauliOperator([[1.0, Pauli(v, ws[i, :])]])
                    for i in range(self._cost_operator.num_qubits)
                ]
            )
        else:
            if not isinstance(mixer_operator, WeightedPauliOperator):
                raise TypeError('The mixer should be a qiskit.aqua.operators.WeightedPauliOperator '
                                + 'object, found {} instead'.format(type(mixer_operator)))
            self._mixer_operator = mixer_operator

    def construct_circuit(self, angles):
        if not len(angles) == self.num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self.num_parameters, len(angles)
            ))
        circuit = QuantumCircuit()
        if self._initial_state:
            circuit += self._initial_state.construct_circuit('circuit')
        if len(circuit.qregs) == 0:
            q = QuantumRegister(self._cost_operator.num_qubits, name='q')
            circuit.add_register(q)
        elif len(circuit.qregs) == 1:
            q = circuit.qregs[0]
        else:
            raise NotImplementedError
        circuit.u2(0, np.pi, q)
        for idx in range(self._p):
            beta, gamma = angles[idx], angles[idx + self._p]
            circuit += self._cost_operator.evolve(
                evo_time=gamma, num_time_slices=1, quantum_registers=q
            )
            circuit += self._mixer_operator.evolve(
                evo_time=beta, num_time_slices=1, quantum_registers=q
            )
        return circuit

    @property
    def setting(self):
        ret = "Variational Form: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

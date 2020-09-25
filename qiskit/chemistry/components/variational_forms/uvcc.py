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
""" Unitary Vibrational Coupled-Cluster Single and Double excitations variational form. """

import logging
import sys
from typing import Optional, List, Tuple, Union, cast

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.circuit import ParameterVector, Parameter

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.chemistry.bosonic_operator import BosonicOperator

logger = logging.getLogger(__name__)


class UVCC(VariationalForm):
    """
    This trial wavefunction is a Unitary Vibrational Coupled-Cluster Single and Double excitations
    variational form.
    For more information, see Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    def __init__(self, num_qubits: int,
                 basis: List[int],
                 degrees: List[int],
                 reps: int = 1,
                 excitations: Optional[List[List[List[int]]]] = None,
                 initial_state: Optional[InitialState] = None,
                 qubit_mapping: str = 'direct',
                 num_time_slices: int = 1,
                 shallow_circuit_concat: bool = True) -> None:
        """

        Args:
            num_qubits: number of qubits
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4]
            degrees: degree of excitation to be included (for single and double excitations
                degrees=[0,1])
            reps: number of replica of basic module
            excitations: The excitations to be included in the circuit.
                If not provided the default is to compute all singles and doubles.
            initial_state: An initial state object.
            qubit_mapping: the qubits mapping type. Only 'direct' is supported at the moment.
            num_time_slices: parameters for dynamics.
            shallow_circuit_concat: indicate whether to use shallow (cheap) mode for
                circuit concatenation
        """

        super().__init__()
        self._num_qubits = num_qubits
        self._num_modes = len(basis)
        self._basis = basis
        self._reps = reps
        self._initial_state = initial_state
        self._qubit_mapping = qubit_mapping
        self._num_time_slices = num_time_slices
        if excitations is None:
            self._excitations = \
                cast(List[List[List[int]]], UVCC.compute_excitation_lists(basis, degrees))
        else:
            self._excitations = excitations

        self._hopping_ops, self._num_parameters = self._build_hopping_operators()
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        self._logging_construct_circuit = True
        self._shallow_circuit_concat = shallow_circuit_concat
        self._support_parameterized_circuit = True

    def _build_hopping_operators(self):
        if logger.isEnabledFor(logging.DEBUG):
            TextProgressBar(sys.stderr)

        results = parallel_map(UVCC._build_hopping_operator, self._excitations,
                               task_args=(self._basis, 'direct'),
                               num_processes=aqua_globals.num_processes)
        hopping_ops = [qubit_op for qubit_op in results if qubit_op is not None]
        num_parameters = len(hopping_ops) * self._reps

        return hopping_ops, num_parameters

    @staticmethod
    def _build_hopping_operator(index: List[List[int]], basis: List[int], qubit_mapping: str) \
            -> WeightedPauliOperator:
        """
        Builds a hopping operator given the list of indices (index) that is a single, a double
        or a higher order excitation.

        Args:
            index: the indexes defining the excitation
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4]
            qubit_mapping: the qubits mapping type. Only 'direct' is supported at the moment.

        Returns:
            Qubit operator object corresponding to the hopping operator

        """

        degree = len(index)
        hml = []  # type: List[List]
        for _ in range(degree):
            hml.append([])

        tmp = []
        tmpdag = []
        for i in range(len(index))[::-1]:
            tmp.append(index[i])
            tmpdag.append([index[i][0], index[i][2], index[i][1]])

        hml[-1].append([tmp, 1])
        hml[-1].append([tmpdag, -1])

        dummpy_op = BosonicOperator(np.asarray(hml, dtype=object), basis)
        qubit_op = dummpy_op.mapping(qubit_mapping)
        if len(qubit_op.paulis) == 0:
            qubit_op = None

        return qubit_op

    @property
    def num_qubits(self) -> int:
        """Number of qubits of the variational form.

        Returns:
           int:  An integer indicating the number of qubits.
        """
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits of the variational form.

        Args:
           num_qubits: An integer indicating the number of qubits.
        """
        self._num_qubits = num_qubits

    def construct_circuit(self, parameters: Union[np.ndarray, List[Parameter], ParameterVector],
                          q: Optional[QuantumRegister] = None) -> QuantumCircuit:
        """Construct the variational form, given its parameters.

        Args:
            parameters: circuit parameters
            q: Quantum Register for the circuit.

        Returns:
            Quantum Circuit a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
            ValueError: if num_qubits has not been set and is still None
        """

        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if self._num_qubits is None:
            raise ValueError('The number of qubits is None and must be set before the circuit '
                             'can be created.')

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        if logger.isEnabledFor(logging.DEBUG) and self._logging_construct_circuit:
            logger.debug("Evolving hopping operators:")
            TextProgressBar(sys.stderr)
            self._logging_construct_circuit = False

        num_excitations = len(self._hopping_ops)

        results = parallel_map(UVCC._construct_circuit_for_one_excited_operator,
                               [(self._hopping_ops[index % num_excitations], parameters[index])
                                for index in range(self._reps * num_excitations)],
                               task_args=(q, self._num_time_slices),
                               num_processes=aqua_globals.num_processes)
        for qc in results:
            if self._shallow_circuit_concat:
                circuit.data += qc.data
            else:
                circuit += qc

        return circuit

    @staticmethod
    def _construct_circuit_for_one_excited_operator(
            qubit_op_and_param: Tuple[WeightedPauliOperator, float],
            qr: QuantumRegister, num_time_slices: int) -> QuantumCircuit:
        """ Construct the circuit building block corresponding to one excitation operator

        Args:
            qubit_op_and_param: list containing the qubit operator and the parameter
            qr: the quantum register to build the circuit on
            num_time_slices: the number of time the building block should be added,
                this should be set to 1

        Returns:
             The quantum circuit
        """
        qubit_op, param = qubit_op_and_param
        qubit_op = qubit_op * -1j
        qc = qubit_op.evolve(state_in=None, evo_time=param, num_time_slices=num_time_slices,
                             quantum_registers=qr)

        return qc

    @staticmethod
    def compute_excitation_lists(basis: List[int], degrees: List[int]) -> List[List[int]]:
        """Compute the list with all possible excitation for given orders

        Args:
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4]
            degrees: degree of excitation to be included (for single and double excitations
                degrees=[0,1])

        Returns:
            List of excitation indexes in terms of modes and modals

        Raises:
            ValueError: If excitation degree is greater than size of basis
        """

        excitation_list = []  # type: List[List[int]]

        def combine_modes(modes, tmp, results, degree):

            if degree >= 0:
                for m, _ in enumerate(modes):
                    combine_modes(modes[m+1:], tmp+[modes[m]], results, degree-1)
            else:
                results.append(tmp)

        def indexes(excitations, results, modes, n, basis):
            if n >= 0:
                for j in range(1, basis[modes[n]]):
                    indexes(excitations + [[modes[n], 0, j]], results, modes, n - 1, basis)
            else:
                results.append(excitations)

        for degree in degrees:
            if degree >= len(basis):
                raise ValueError('The degree of excitation cannot be '
                                 'greater than the number of modes')

            combined_modes = []  # type: List
            modes = []
            for i in range(len(basis)):
                modes.append(i)

            combine_modes(modes, [], combined_modes, degree)

            for element in combined_modes:
                indexes([], excitation_list, element, len(element)-1, basis)

        return excitation_list

    def excitations_in_qubit_format(self) -> List[List[int]]:
        """Gives the list of excitation indexes in terms of qubit indexes rather
         than in modes and modals

        Returns:
            List of excitation indexes

        """

        result = []

        for excitation in self._excitations:

            dummy_ex = []
            for element in excitation:
                q_count = 0
                for idx in range(element[0]):
                    q_count += self._basis[idx]

                dummy_ex.append(q_count+element[1])
                dummy_ex.append(q_count+element[2])

            dummy_ex.sort()
            result.append(dummy_ex)

        return result

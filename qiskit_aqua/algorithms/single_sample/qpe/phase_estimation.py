# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
Quantum Phase Estimation.
"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aqua import Operator, AquaError


class PhaseEstimation:

    def __init__(
            self, operator, state_in, iqft,
            num_time_slices=1,
            num_ancillae=1,
            paulis_grouping='random',
            expansion_mode='trotter',
            expansion_order=1,
            state_in_circuit_factory=None,
            unitary_circuit_factory=None,
            shallow_circuit_concat=False):
        """
        Constructor.

        Args:
            operator (Operator): the hamiltonian Operator object
            state_in (InitialState): the InitialState pluggable component representing the initial quantum state
            iqft (IQFT): the Inverse Quantum Fourier Transform pluggable component
            num_time_slices (int): the number of time slices
            num_ancillae (int): the number of ancillary qubits to use for the measurement
            paulis_grouping (str): the pauli term grouping mode
            expansion_mode (str): the expansion mode (trotter|suzuki)
            expansion_order (int): the suzuki expansion order
            state_in_circuit_factory (CircuitFactory): the initial state represented by a CircuitFactory object
            unitary_circuit_factory (CircuitFactory): the problem unitary represented by a CircuitFactory object
            shallow_circuit_concat (bool): indicate whether to use shallow (cheap) mode for circuit concatenation
        """

        if (
                operator is not None and unitary_circuit_factory is not None
        ) or (
                operator is None and unitary_circuit_factory is None
        ):
            raise AquaError('Please supply either an operator or a unitary circuit factory but not both.')

        self._operator = operator
        self._unitary_circuit_factory = unitary_circuit_factory
        self._state_in = state_in
        self._state_in_circuit_factory = state_in_circuit_factory
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._paulis_grouping = paulis_grouping
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._shallow_circuit_concat = shallow_circuit_concat
        self._ancilla_phase_coef = 1
        self._circuit = {True: None, False: None}
        self._ret = {}

    def construct_circuit(self, state_register=None, ancilla_register=None, aux_register=None, measure=False):
        """
        Construct the Phase Estimation circuit

        Args:
            state_register (QuantumRegister): the optional register to use for the quantum state
            ancilla_register (QuantumRegister): the optional register to use for the ancillary measurement qubits
            aux_register (QuantumRegister): an optional auxiliary quantum register
            measure (bool): boolean flag to indicate if the built circuit should include ancilla measurement

        Returns:
            the QuantumCircuit object for the constructed circuit
        """

        if self._circuit[measure] is None:
            if self._operator is not None:
                # check for identify paulis to get its coef for applying global phase shift on ancillae later
                num_identities = 0
                for p in self._operator.paulis:
                    if np.all(np.logical_not(p[1].z)) and np.all(np.logical_not(p[1].x)):
                        num_identities += 1
                        if num_identities > 1:
                            raise RuntimeError('Multiple identity pauli terms are present.')
                        self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

            if ancilla_register is None:
                a = QuantumRegister(self._num_ancillae, name='a')
            else:
                a = ancilla_register

            if state_register is None:
                if self._operator is not None:
                    q = QuantumRegister(self._operator.num_qubits, name='q')
                elif self._unitary_circuit_factory is not None:
                    q = QuantumRegister(self._unitary_circuit_factory.num_target_qubits, name='q')
                else:
                    raise RuntimeError('Missing operator specification.')
            else:
                q = state_register
            qc = QuantumCircuit(a, q)

            if aux_register is None:
                num_aux_qubits, aux = 0, None
                if self._state_in_circuit_factory is not None:
                    num_aux_qubits = self._state_in_circuit_factory.required_ancillas()
                if self._unitary_circuit_factory is not None:
                    num_aux_qubits = max(num_aux_qubits, self._unitary_circuit_factory.required_ancillas_controlled())

                if num_aux_qubits > 0:
                    aux = QuantumRegister(num_aux_qubits, name='aux')
                    qc.add_register(aux)
            else:
                aux = aux_register
                qc.add_register(aux)

            # initialize state_in
            if self._state_in is not None:
                qc.data += self._state_in.construct_circuit('circuit', q).data
            elif self._state_in_circuit_factory is not None:
                self._state_in_circuit_factory.build(qc, q, aux)
            else:
                raise RuntimeError('Missing initial state specification.')

            # Put all ancillae in uniform superposition
            qc.u2(0, np.pi, a)

            # phase kickbacks via dynamics
            if self._operator is not None:
                pauli_list = self._operator.reorder_paulis(grouping=self._paulis_grouping)
                if len(pauli_list) == 1:
                    slice_pauli_list = pauli_list
                else:
                    if self._expansion_mode == 'trotter':
                        slice_pauli_list = pauli_list
                    elif self._expansion_mode == 'suzuki':
                        slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(
                            pauli_list,
                            1,
                            self._expansion_order
                        )
                    else:
                        raise ValueError('Unrecognized expansion mode {}.'.format(self._expansion_mode))
                for i in range(self._num_ancillae):
                    qc_evolutions = Operator.construct_evolution_circuit(
                        slice_pauli_list, -2 * np.pi, self._num_time_slices, q, a, ctl_idx=i,
                        shallow_slicing=self._shallow_circuit_concat
                    )
                    if self._shallow_circuit_concat:
                        qc.data += qc_evolutions.data
                    else:
                        qc += qc_evolutions
                    # global phase shift for the ancilla due to the identity pauli term
                    qc.u1(2 * np.pi * self._ancilla_phase_coef * (2 ** i), a[i])
            elif self._unitary_circuit_factory is not None:
                for i in range(self._num_ancillae):
                    self._unitary_circuit_factory.build_controlled_power(qc, q, a[i], 2 ** i, aux)

            # inverse qft on ancillae
            self._iqft.construct_circuit('circuit', a, qc)

            # measuring ancillae
            if measure:
                c = ClassicalRegister(self._num_ancillae, name='c')
                qc.add_register(c)
                qc.barrier(a)
                qc.measure(a, c)

            self._circuit[measure] = qc

        return self._circuit[measure]

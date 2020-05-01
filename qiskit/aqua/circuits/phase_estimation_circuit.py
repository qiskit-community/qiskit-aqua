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
"""
Quantum Phase Estimation Circuit.
"""

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

from qiskit.aqua import AquaError
from qiskit.aqua.operators import (WeightedPauliOperator,   # pylint: disable=unused-import
                                   suzuki_expansion_slice_pauli_list,
                                   evolution_instruction)


class PhaseEstimationCircuit:
    """
    Quantum Phase Estimation Circuit.
    """
    def __init__(
            self,
            operator=None,
            state_in=None,
            iqft=None,
            num_time_slices=1,
            num_ancillae=1,
            expansion_mode='trotter',
            expansion_order=1,
            evo_time=2 * np.pi,
            state_in_circuit_factory=None,
            unitary_circuit_factory=None,
            shallow_circuit_concat=False,
            pauli_list=None
    ):
        """
        Constructor.

        Args:
            operator (WeightedPauliOperator): the hamiltonian Operator object
            state_in (InitialState): the InitialState pluggable component
            representing the initial quantum state
            iqft (IQFT): the Inverse Quantum Fourier Transform pluggable component
            num_time_slices (int): the number of time slices
            num_ancillae (int): the number of ancillary qubits to use for the measurement
            expansion_mode (str): the expansion mode (trotter|suzuki)
            expansion_order (int): the suzuki expansion order
            evo_time (float): the evolution time
            state_in_circuit_factory (CircuitFactory): the initial state represented by
            a CircuitFactory object
            unitary_circuit_factory (CircuitFactory): the problem unitary represented
            by a CircuitFactory object
            shallow_circuit_concat (bool): indicate whether to use shallow (cheap) mode
            for circuit concatenation
            pauli_list (list[Pauli]): the flat list of paulis for the operator
        Raises:
            AquaError: Missing input
        """

        if (operator is not None and unitary_circuit_factory is not None) or \
                (operator is None and unitary_circuit_factory is None):
            raise AquaError('Please supply either an operator or a unitary circuit '
                            'factory but not both.')

        self._operator = operator
        if operator is not None:
            self._pauli_list = operator.reorder_paulis() if pauli_list is None else pauli_list
        self._unitary_circuit_factory = unitary_circuit_factory
        self._state_in = state_in
        self._state_in_circuit_factory = state_in_circuit_factory
        self._iqft = iqft
        self._num_time_slices = num_time_slices
        self._num_ancillae = num_ancillae
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._evo_time = evo_time
        self._shallow_circuit_concat = shallow_circuit_concat
        self._ancilla_phase_coef = 1
        self._state_register = None
        self._ancillary_register = None
        self._auxiliary_register = None
        self._circuit = None

    def construct_circuit(
            self,
            state_register=None,
            ancillary_register=None,
            auxiliary_register=None,
            measurement=False,
    ):
        """
        Construct the Phase Estimation circuit

        Args:
            state_register (QuantumRegister): the optional register to use for the quantum state
            ancillary_register (QuantumRegister): the optional register to use for
            the ancillary measurement qubits
            auxiliary_register (QuantumRegister): an optional auxiliary quantum register
            measurement (bool): Boolean flag to indicate if measurement should be included
            in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        Raises:
            RuntimeError: Multiple identity pauli terms are present
            ValueError: invalid mode
        """
        # pylint: disable=invalid-name
        if self._circuit is None:
            if ancillary_register is None:
                a = QuantumRegister(self._num_ancillae, name='a')
            else:
                a = ancillary_register
            self._ancillary_register = a

            if state_register is None:
                if self._operator is not None:
                    q = QuantumRegister(self._operator.num_qubits, name='q')
                elif self._unitary_circuit_factory is not None:
                    q = QuantumRegister(self._unitary_circuit_factory.num_target_qubits, name='q')
                else:
                    raise RuntimeError('Missing operator specification.')
            else:
                q = state_register
            self._state_register = q

            qc = QuantumCircuit(a, q)

            if auxiliary_register is None:
                num_aux_qubits, aux = 0, None
                if self._state_in_circuit_factory is not None:
                    num_aux_qubits = self._state_in_circuit_factory.required_ancillas()
                if self._unitary_circuit_factory is not None:
                    num_aux_qubits = \
                        max(num_aux_qubits,
                            self._unitary_circuit_factory.required_ancillas_controlled())

                if num_aux_qubits > 0:
                    aux = QuantumRegister(num_aux_qubits, name='aux')
                    qc.add_register(aux)
            else:
                aux = auxiliary_register
                qc.add_register(aux)

            # initialize state_in
            if self._state_in is not None:
                qc.data += self._state_in.construct_circuit('circuit', q).data
            elif self._state_in_circuit_factory is not None:
                self._state_in_circuit_factory.build(qc, q, aux)

            # Put all ancillae in uniform superposition
            qc.u2(0, np.pi, a)

            # phase kickbacks via dynamics
            if self._operator is not None:
                # check for identify paulis to get its coef for applying
                # global phase shift on ancillae later
                num_identities = 0
                for p in self._pauli_list:
                    if np.all(np.logical_not(p[1].z)) and np.all(np.logical_not(p[1].x)):
                        num_identities += 1
                        if num_identities > 1:
                            raise RuntimeError('Multiple identity pauli terms are present.')
                        self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

                if len(self._pauli_list) == 1:
                    slice_pauli_list = self._pauli_list
                else:
                    if self._expansion_mode == 'trotter':
                        slice_pauli_list = self._pauli_list
                    elif self._expansion_mode == 'suzuki':
                        slice_pauli_list = suzuki_expansion_slice_pauli_list(
                            self._pauli_list,
                            1,
                            self._expansion_order
                        )
                    else:
                        raise ValueError(
                            'Unrecognized expansion mode {}.'.format(self._expansion_mode))
                for i in range(self._num_ancillae):

                    qc_evolutions_inst = evolution_instruction(
                        slice_pauli_list, -self._evo_time,
                        self._num_time_slices, controlled=True, power=(2 ** i),
                        shallow_slicing=self._shallow_circuit_concat)
                    if self._shallow_circuit_concat:
                        qc_evolutions = QuantumCircuit(q, a)
                        qc_evolutions.append(qc_evolutions_inst, qargs=list(q) + [a[i]])
                        qc.data += qc_evolutions.data
                    else:
                        qc.append(qc_evolutions_inst, qargs=list(q) + [a[i]])
                    # global phase shift for the ancilla due to the identity pauli term
                    qc.u1(self._evo_time * self._ancilla_phase_coef * (2 ** i), a[i])

            elif self._unitary_circuit_factory is not None:
                for i in range(self._num_ancillae):
                    self._unitary_circuit_factory.build_controlled_power(qc, q, a[i], 2 ** i, aux)

            # inverse qft on ancillae
            self._iqft.construct_circuit(mode='circuit', qubits=a, circuit=qc, do_swaps=False)

            if measurement:
                c_ancilla = ClassicalRegister(self._num_ancillae, name='ca')
                qc.add_register(c_ancilla)
                # qc.barrier(a)
                qc.measure(a, c_ancilla)

            self._circuit = qc
        return self._circuit

    @property
    def state_register(self):
        """ returns state register """
        return self._state_register

    @property
    def ancillary_register(self):
        """ returns ancillary register """
        return self._ancillary_register

    @property
    def auxiliary_register(self):
        """ returns auxiliary register """
        return self._auxiliary_register

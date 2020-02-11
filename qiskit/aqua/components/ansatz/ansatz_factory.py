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

"""The Ansatz class."""

from typing import Union, Optional, List

import numbers
import numpy
from qiskit import QuantumCircuit, QiskitError, transpile
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector
from qiskit.aqua import AquaError


class Ansatz:
    """The Ansatz class.

    TODO
    """

    def __init__(self,
                 gates: Optional[Union[Gate, List[Gate]]] = None,
                 qubit_indices: Optional[Union[List[int], List[List[int]]]] = None,
                 reps: Optional[Union[int, List[int]]] = None,
                 insert_barriers: bool = False) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            gates: The input gates. Can be a single gate, a list of gates, (or circuits?)
            qubit_indices: The indices specifying on which qubits the input gates act. If None, for
                each gate this is set to the first `n` qubits, where `n` is the number of qubits the
                gate acts on.
            reps: Specifies how the input gates are repeated. If an integer, all input gates
                are repeated `reps` times (in the provided order). If a list of
                integers, `reps` determines the order of the layers in Ansatz using the elements
                of `reps` as index. See the Examples section for more detail.
            insert_barriers: If True, barriers are inserted in between each layer/gate. If False,
                no barriers are inserted.

        Examples:
            todo
        """

        # get gates in the right format
        if isinstance(gates, Gate):  # convert gates to a list if necessary (or use hasattr)
            self._gates = [gates]
        elif gates is None:
            self._gates = []

        # get reps in the right format
        if reps is None:  # if reps is None, set it to [0, .., len(num_gates) - 1]
            self._reps = list(range(len(self._gates)))
        elif isinstance(reps, int):  # if reps is an int, set it to reps * [0, ..., len(gates) - 1]
            self._reps = reps * list(range(len(self._gates)))
        else:  # right format
            self._reps = reps

        # get qubit_indices in the right format (i.e. list of lists)
        if qubit_indices is None:
            self._qargs = [list(range(gate.num_qubits)) for gate in self._gates]
        elif not isinstance(qubit_indices[0], list):
            self._qargs = [qubit_indices]
        else:  # right format
            self._qargs = qubit_indices

        # maximum number of qubits
        self._num_qubits = numpy.max(self._qargs) if len(self._qargs) > 0 else 0

        # insert barriers?
        self._insert_barriers = insert_barriers

        # keep track of the circuit
        self._circuit = None

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this Ansatz.

        Returns:
            The number of qubits.
        """
        return self._num_qubits

    @property
    def params(self) -> Union[List[float], List[Parameter]]:
        """Get the parameters of the Ansatz.

        Returns:
            A list containing the parameters.
        """
        if self._circuit is None:
            return []

        return list(self._circuit.parameters)

    @params.setter
    def params(self, params: Union[List[float], List[Parameter], ParameterVector]) -> None:
        """Set the parameters of the Ansatz.

        Args:
            The new parameters.

        Raises:
            ValueError: If the number of provided parameters does not match the number of
                parameters of the Ansatz.
            TypeError: If the type of `params` is not supported.
        """
        if len(params) != len(self.params):
            raise ValueError('Mismatching number of parameters!')

        # if the provided parameters are real values, bind them
        if all(isinstance(param, numbers.Real) for param in params):
            param_dict = dict(zip(self.params, params))
            self._circuit = self._circuit.bind_parameters(param_dict)

        # if they are new parameters, replace them in the circuit
        elif all(isinstance(param, Parameter) for param in params) \
                or isinstance(params, ParameterVector):
            param_dict = dict(zip(self.params, params))
            self._circuit._substitute_parameters(param_dict)

        # otherwise the input type is not supported
        else:
            raise TypeError('Unsupported type of `params`.')

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in the Ansatz.

        Returns:
            The number of parameters.
        """
        return len(self.params)

    def to_circuit(self) -> QuantumCircuit:
        """Convert the Ansatz into a circuit.

        If the Ansatz has not been defined, an empty quantum circuit is returned.

        Returns:
            A quantum circuit containing this Ansatz. The width of the circuit equals
            the number of qubits in this Ansatz.
        """
        # build the circuit if it has not been constructed yet
        if self._circuit is None:
            if self.num_qubits == 0:
                circuit = QuantumCircuit()

            else:
                circuit = QuantumCircuit(self._num_qubits)

                # add the gates, if they are specified
                if len(self._reps) > 0:
                    # the first gate (separately so barriers can be inserted in the for-loop)
                    idx = self._reps[0]
                    circuit.append(self._gates[idx], self._qargs[idx])

                    for idx in self._reps[1:]:
                        if self._insert_barriers:  # insert barrier, if necessary
                            circuit.barrier()
                        circuit.append(self._gates[idx], self._qargs[idx])  # add next layer

                # store the circuit
            self._circuit = circuit

        return self._circuit

    def __repr__(self) -> str:
        """Draw this Ansatz in circuit format using the standard gates.

        Returns:
            A single string representing this Ansatz.
        """
        basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz',
                       'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap', 'cswap',
                       'toffoli', 'u1', 'u2', 'u3']
        return transpile(self.to_circuit(), basis_gates=basis_gates).draw().single_string()

    @property
    def insert_barriers(self) -> bool:
        """Check whether the Ansatz inserts barriers or not.

        Returns:
            True, if barriers are inserted in between the layers, False if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers should be inserted in between the layers or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        # if insert_barriers changes, we have to invalide the circuit definition,
        # if it is the same as before we can leave the Ansatz instance as it is
        if insert_barriers is not self._insert_barriers:
            self._circuit = None
            self._insert_barriers = insert_barriers

    def to_instruction(self) -> Instruction:
        """Convert the Ansatz into an Instruction.

        Returns:
            An Instruction containing this Ansatz.
        """
        return self.to_circuit().to_instruction()

    def to_gate(self) -> Gate:
        """Convert this Ansatz into a Gate, if possible.

        If the Ansatz contains only unitary operations (i.e. neither measurements nor barriers)
        return this Ansatz as a Gate.

        Returns:
            A Gate containing this Ansatz.

        Raises:
            AquaError: If the Ansatz contains non-unitary operations.
        """
        try:
            return self.to_circuit().to_gate()
        except QiskitError:
            raise AquaError('The Ansatz contains non-unitary operations (e.g. barriers or '
                            'measurements) and cannot be converted to a Gate!')

    def append(self, gate: Gate, qubit_indices: Optional[List[int]] = None) -> None:
        """Append another gate to the Ansatz.

        Args:
            gate: The gate to append.
            qubit_indices: The qubit indices where to append the gate to.
                Defaults to the first `n` qubits, where `n` is the number of qubits the gate acts
                on.

        Note:
            TODO make it work if the width of gate exceeds the current num qubits
        """
        # keep track of the input
        self._gates += [gate]
        self._reps += [len(self._gates) - 1]

        # We can have two cases: the appended gate fits onto the current Ansatz (i.e. has
        # less of equal number of qubits), or exceeds the number of qubits.
        # In the latter case we have to add an according offset to the qubit indices.

        self._qargs += [qubit_indices or list(range(gate.num_qubits))]
        print('qargs:', self._qargs[-1])

        # modify the circuit accordingly
        if self._circuit is None:
            _ = self.to_circuit()  # automatically constructed
        else:
            self._circuit.append(self._gates[-1], self._qargs[-1], [])  # append gate

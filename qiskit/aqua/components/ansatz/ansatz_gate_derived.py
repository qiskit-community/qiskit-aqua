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
import copy
from typing import Union, Optional, List, Tuple

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class Ansatz(Gate):
    """The Ansatz class.

    TODO
    """

    def __init__(self,
                 gates: Union[Gate, List[Gate]],
                 qubit_indices: Optional[Union[List[int], List[List[int]]]],
                 reps: Optional[Union[int, List[int]]],
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

        # get reps in the right format
        if reps is None:  # if reps is None, set it to [0, .., len(num_gates) - 1]
            self._reps = list(range(len(gates)))
        elif isinstance(reps, int):  # if reps is an int, set it to reps * [0, ..., len(gates) - 1]
            self._reps = reps * list(range(len(gates)))
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
        self._num_qubits = numpy.max(self._qargs)

        super().__init__('some_name_to_be_figured_out', self._num_qubits, [])

        # insert barriers?
        self._insert_barriers = insert_barriers

        # lazily define
        self._definition = None

    @property
    def params(self):
        pass

    def _define(self):
        """Defines the Ansatz using the internal variables."""

        # use a circuit to append all the gates
        circuit = QuantumCircuit(self._num_qubits)

        # the first gate (separately so barriers can be inserted in the for-loop)
        idx = self._reps[0]
        circuit.append(self._gates[idx], self._qargs[idx])

        for idx in self._reps[1:]:
            if self._insert_barriers:  # insert barrier, if necessary
                circuit.barrier()
            circuit.append(self._gates[idx], self._qargs[idx])  # add next layer

        self._definition = circuit.data

    def to_circuit(self) -> QuantumCircuit:
        """Convert the Ansatz into a circuit.

        If the Ansatz has not been defined, an empty quantum circuit is returned.

        Returns:
            A quantum circuit containing this Ansatz. The width of the circuit equals
            the number of qubits in this Ansatz.
        """

        if self.definition is None:
            return QuantumCircuit()  # return an empty circuit if no definition was found

        circuit = QuantumCircuit(self.num_qubits)
        circuit.append(self, list(range(self.num_qubits)), [])
        return circuit

    def append(self, gate, qubit_indices=None):
        """Append another gate to the Ansatz.

        Args:
            gate: The gate to append.
            qubit_indices: The qubit indices where to append the gate to.
                Defaults to the first `n` qubits, where `n` is the number of qubits the gate acts
                on.
        """
        self._definition = None  # invalidate definition
        self._gates += [gate]
        self._qargs += [qubit_indices or list(range(gate.num_qubits))]
        self._reps += [len(self._gates) - 1]

    def etc(self):
        pass

    def copy(self, name=None):
        """Get a copy of self. Can be used to append a copy of self to a QuantumCircuit."""
        copied_ansatz = copy.deepcopy(self)
        if name is not None:
            copied_ansatz.name = name

        return copied_ansatz

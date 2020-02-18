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

"""The Ansatz class.

TODO
    * store ccts instead of gates?
        - Reverting to ccts in future anyways
    * add transpile feature?
    * add params argument to to_circuit?
    * copy input layers? probably we should
    * keep a separate list of parameters to ensure the same order and be able to change the
        parameters w/o constructing the cct
"""

from __future__ import annotations  # to use the type hint 'Ansatz' in the class itself
from typing import Union, Optional, List, Any

import numbers
import numpy
from qiskit import QuantumCircuit, QiskitError, transpile
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import InitialState


class Ansatz:
    """The Ansatz class.

    Attributes:
        blocks: The single building blocks of the Ansatz.
        params: The parameters of the Ansatz.
    """

    def __init__(self,
                 blocks: Optional[Union[Gate, List[Gate]]] = None,
                 qubit_indices: Optional[Union[List[int], List[List[int]]]] = None,
                 reps: Optional[Union[int, List[int]]] = None,
                 insert_barriers: bool = False,
                 initial_state: Optional[InitialState] = None) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            blocks: The input blocks. Can be a single gate, a list of gates, (or circuits?)
            qubit_indices: The indices specifying on which qubits the input blocks act. If None, for
                each block this is set to the first `n` qubits, where `n` is the number of qubits
                the block acts on.
            reps: Specifies how the input blocks are repeated. If an integer, all input blocks
                are repeated `reps` times (in the provided order). If a list of
                integers, `reps` determines the order of the layers in Ansatz using the elements
                of `reps` as index. See the Examples section for more detail.
            insert_barriers: If True, barriers are inserted in between each layer/block. If False,
                no barriers are inserted.
            initial_state: An `InitialState` object to prepent to the Ansatz.
                TODO deprecate this feature in favour of prepend or overloading __add__ in
                the initial state class

        Raises:
            TypeError: If `blocks` contains an unsupported object.

        Examples:
            todo
        """
        # insert barriers?
        self._insert_barriers = insert_barriers

        # get blocks in the right format
        if blocks is None:
            blocks = []

        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._blocks = []
        for block in blocks:
            self._blocks += [self._convert_to_block(block)]

        # get reps in the right format
        if reps is None:  # if reps is None, set it to [0, .., len(num_blocks) - 1]
            self._reps = list(range(len(self._blocks)))
        elif isinstance(reps, int):  # if reps is an int, set it to reps * [0, ..., len(blocks) - 1]
            self._reps = reps * list(range(len(self._blocks)))
        else:  # right format
            self._reps = reps

        # get qubit_indices in the right format (i.e. list of lists)
        if qubit_indices is None:
            self._qargs = [list(range(block.num_qubits)) for block in self._blocks]
        elif not isinstance(qubit_indices[0], list):
            self._qargs = [qubit_indices]
        else:  # right format
            self._qargs = qubit_indices

        # maximum number of qubits
        self._num_qubits = int(numpy.max(self._qargs) + 1 if len(self._qargs) > 0 else 0)

        # if there is an initial state object, check that the number of qubits is compatible
        # construct the circuit immediately since we need the number of qubits
        # alternate solution: add num_qubits as attribute to the InitialState
        self._initial_state_circuit = None
        if initial_state is not None:
            # construct the circuit
            self._initial_state_circuit = initial_state.construct_circuit(mode='circuit')

            # the initial state dictates the number of qubits since we do not have information
            # about on which qubits the initial state acts
            if self._initial_state_circuit.n_qubits < self._num_qubits:
                raise ValueError('The provided initial state has less qubits than the Ansatz.')

            self._num_qubits = self._initial_state_circuit.n_qubits > self._num_qubits

        # keep track of the circuit
        self._circuit = None

    def _convert_to_block(self, layer: Any) -> Instruction:
        """Try to convert `layer` to an Instruction.

        Args:
            layer: The object to be converted to an Ansatz block / Instruction.

        Raises:
            TypeError: If the input cannot be converted to an Instruction.

        Returns:
            The layer converted to an Instruction.
        """
        if isinstance(layer, Instruction):
            return layer
        elif hasattr(layer, 'to_instruction'):
            return layer.to_instruction()
        else:
            raise TypeError('Adding a {} to an Ansatz is not supported.'.format(type(layer)))

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this Ansatz.

        Returns:
            The number of qubits.
        """
        return int(self._num_qubits)

    @property
    def params(self) -> Union[List[float], List[Parameter]]:
        """Get the parameters of the Ansatz.

        Returns:
            A list containing the parameters.
        """
        return list(self.to_circuit().parameters)

    @params.setter
    def params(self, params: Union[dict, List[float], List[Parameter], ParameterVector]) -> None:
        """Set the parameters of the Ansatz.

        Args:
            The new parameters.

        Raises:
            ValueError: If the number of provided parameters does not match the number of
                parameters of the Ansatz.
            TypeError: If the type of `params` is not supported.
        """
        # TODO figure out whether it is more efficient to iterate over the list and check for
        # values in the dictionary, or iterate over the dictionary and find the according value
        # in the list. Random access via element should be much faster in the dictionary, probably.
        if isinstance(params, dict):
            new_params = []
            for i, current_param in enumerate(self.params):
                # try to get the new value, if there is none, use the current value
                new_params[i] = params.get(current_param, self.params[i])
        else:
            if len(params) != len(self.params):
                raise ValueError('Mismatching number of parameters!')
            new_params = params

        if self._circuit is not None:
            self._bind_circuit_parameters(new_params)

    def _bind_circuit_parameters(self, params):
        """Bind the params to the underlying circuit."""
        if all(isinstance(param, numbers.Real) for param in params):
            param_dict = dict(zip(self.params, params))
            self._circuit = self._circuit.bind_parameters(param_dict)

        # if they are new parameters, replace them in the circuit
        elif all(isinstance(param, Parameter) for param in params):
            param_dict = dict(zip(self.params, params))
            self._circuit._substitute_parameters(param_dict)

        # otherwise the input type is not supported
        else:
            raise TypeError('Unsupported type of `params`, {}'.type(params))

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in the Ansatz.

        Returns:
            The number of parameters.
        """
        return len(self.params)

    def construct_circuit(self, params: Union[List[float], List[Parameter], ParameterVector]
                          ) -> QuantumCircuit:
        """Deprecated, use `to_circuit()` -- supporting backward compatibility.

        Args:
            params: The parameters for the Ansatz.

        Returns:
            The Ansatz as circuit.
        """
        self.params = params
        return self.to_circuit()

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
                # use the initial state circuit if it is not None
                circuit = self._initial_state_circuit or QuantumCircuit(self.num_qubits)

                # add the blocks, if they are specified
                if len(self._reps) > 0:
                    # the first block (separately so barriers can be inserted in the for-loop)
                    idx = self._reps[0]
                    circuit.append(self._blocks[idx], self._qargs[idx])

                    for idx in self._reps[1:]:
                        if self._insert_barriers:  # insert barrier, if necessary
                            circuit.barrier()
                        circuit.append(self._blocks[idx], self._qargs[idx])  # add next layer

            # store the circuit
            self._circuit = circuit

        return self._circuit

    def __add__(self, other: Union[Ansatz, Instruction, QuantumCircuit]) -> Ansatz:
        """Overloading + for convenience.

        This presumes list(range(other.num_qubits)) as qubit indices and calls self.append().

        Args:
            other: The object to append.

        Raises:
            TypeError: If the added type is unsupported.

        Returns:
            self
        """
        return self.append(other)

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
        # if insert_barriers changes, we have to invalidate the circuit definition,
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

        If the Ansatz contains only unitary operations(i.e. neither measurements nor barriers)
        return this Ansatz as a Gate.

        Returns:
            A Gate containing this Ansatz.

        Raises:
            AquaError: If the Ansatz contains non-unitary operations.
        """
        try:
            return self.to_circuit().to_gate()
        except QiskitError:
            raise AquaError('The Ansatz contains non-unitary operations (e.g. barriers, resets or '
                            'measurements) and cannot be converted to a Gate!')

    def append(self,
               other: Union[Ansatz, Instruction, QuantumCircuit],
               qubit_indices: Optional[List[int]] = None
               ) -> Ansatz:
        """Append another layer to the Ansatz.

        Args:
            other: The layer to append, can be another Ansatz, an Instruction(hence also a Gate),
                or a QuantumCircuit.
            qubit_indices: The qubit indices where to append the layer to.
                Defaults to the first `n` qubits, where `n` is the number of qubits the layer acts
                on.

        Returns:
            self, such that chained appends are possible.

        Raises:
            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a
                `to_instruction` method.
        """
        # add other to the list of blocks
        block = self._convert_to_block(other)
        self._blocks += [block]
        self._params += block.params

        # keep track of which blocks to add to the Ansatz
        self._reps += [len(self._blocks) - 1]

        # define the the qubit indices
        self._qargs += [qubit_indices or list(range(self._blocks[-1].num_qubits))]

        # retrieve number of qubits
        num_qubits = max(self._qargs[-1]) + 1

        # We can have two cases: the appended block fits onto the current Ansatz (i.e. has
        # less of equal number of qubits), or exceeds the number of qubits.
        # In the latter case we have to add an according offset to the qubit indices.
        # Since we cannot append a circuit of larger size to an existing circuit we have to rebuild
        if num_qubits > self.num_qubits:
            self._num_qubits = num_qubits
            self._circuit = None  # rebuild circuit

        # modify the circuit accordingly
        if self._circuit is None:
            _ = self.to_circuit()  # automatically constructed
        else:
            if self._insert_barriers and len(self._reps) > 1:
                self._circuit.barrier()
            self._circuit.append(self._blocks[-1], self._qargs[-1], [])  # append block

        return self

#
# Experimental parameter setting functionality
#
# def _bind_instruction_parameters(instruction, new_parameters):
#     basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz',
#                    'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap', 'cswap',
#                    'toffoli', 'u1', 'u2', 'u3']
#
#     instruction_params = [p for p in instruction.params if isinstance(p, ParameterExpression)]
#
#     if not isinstance(new_parameters, dict):
#         replacement_table = dict(zip(instruction_params, new_parameters))
#     else:
#         replacement_table = new_parameters
#
#     ops = [(instruction, None, None)]
#     decomposed = []
#     while len(ops) > 0:
#         op = ops.pop()[0]  # returns and removes the last element
#         if op.name in ['barrier', 'reset', 'measure']:
#             continue
#         elif op.name in basis_gates:
#             decomposed += [op]
#         else:
#             ops += op.definition
#
#     for op in decomposed:
#         if len(op.params) > 0:
#             for param in op.params:
#                 if param in replacement_table.keys():
#                     replacement = replacement_table[param]
#                     if isinstance(replacement, ParameterExpression):
#                         param.subs({param: replacement})
#                     else:
#                         param.bind({param: replacement})
#
#
# def _bind_circuit_parameters(circuit, params, experimental):
#     """Propagates the changes down to the circuit."""
#     count = 0
#     for block, _, _ in circuit.data:
#         if isinstance(params, dict):
#             block_params = params
#         else:
#             block_params = params[count:count + len(block.params)]
#             count += len(block_params)
#         _bind_instruction_parameters(block, block_params)
#
#
# def _extract_parameter_list(circuit: QuantumCircuit) -> List[Parameter]:
#     """Fully unroll the circuit and extract a list of every single parameter.
#
#     Args:
#         circuit: The circuit from which to extract the parameter list from.
#
#     Returns:
#         A list of all parameters, including duplicates.
#     """
#
#     basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz',
#                    'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap', 'cswap',
#                    'toffoli', 'u1', 'u2', 'u3']
#     decomposed = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
#
#     params = []
#     for instruction, _, _ in decomposed.data:
#         if len(instruction.params) > 0:
#             params += instruction.params
#
#     return params
#
#
# def _inject_parameter_list(circuit: QuantumCircuit,
#                            parameters: Union[List[float], List[Parameter], ParameterVector]
#                            ) -> None:
#     """Set the parameters."""
#
#     basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz',
#                    'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap', 'cswap',
#                    'toffoli', 'u1', 'u2', 'u3']
#     decomposed = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
#
#     counter = 0
#     for instruction, _, _ in decomposed.data:
#         if len(instruction.params) > 0:
#             instruction.params = [parameters[counter]]
#             counter += 1
#
#     return decomposed

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
        - Performance difference?
    * add transpile feature
    * add params argument to to_circuit?
    * rename append to combine(after=True) with to support after/before
"""

from __future__ import annotations  # to use the type hint 'Ansatz' in the class itself
from typing import Union, Optional, List, Any, Tuple

import numbers
import numpy
from qiskit import QuantumCircuit, QiskitError, transpile, QuantumRegister
from qiskit.circuit import Gate, Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import InitialState


def get_parameters(block: Union[QuantumCircuit, Instruction]) -> List[Parameter]:
    """Return the list of Parameters inside block."""
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]


class Ansatz:
    """The Ansatz class.

    Attributes:
        blocks: The single building blocks of the Ansatz.
        params: The parameters of the Ansatz.
        num_qubits: The number of qubits in the Ansatz.
        num_parameters: The number of tuneable parameters in the Ansatz.
        setting: Returns information about the class properties. To be deprecated.
        support_parameterized_circuit: If True, the parameters can be set with Parameter objects.
            To be deprecated, since it always should be True.
        parameter_bounds: Suggested parameter bounds for the parameters that an optimizer may use.
        preferred_init_points: The Ansatz can store preferred initial points for the paramter
            values. To be deprecated?

    """

    def __init__(self,
                 blocks: Optional[Union[QuantumCircuit, List[QuantumCircuit],
                                        Instruction, List[Instruction]]] = None,
                 qubit_indices: Optional[Union[List[int], List[List[int]]]] = None,
                 reps: Optional[Union[int, List[int]]] = None,
                 insert_barriers: bool = False,
                 blockwise_parameters: Optional[List[Parameter]] = None,
                 initial_state: Optional[InitialState] = None) -> None:
        """Initializer. Constructs the blocks of the Ansatz from the input.

        The structure of the Ansatz are repeated parameterized circuit-blocks (referred to as blocks
        in the code). For every block, qubit indices indicate on which qubits the block acts on.
        Which and how blocks are repeated is determined via the ``reps`` argument.
        If the same block is supposed to be repeated several times with the same qubit indices,
        it is only stored once.  On circuit construction, copies of the blocks are inserted in the
        Ansatz with new, unique parameters.
        These "base parameters" do not change. The user modifies "surface parameters" which are
        bound to the circuit upon requesting the circuit.
        If specified, barriers can be inserted in between every block.
        If an initial state is provided, it is prepended to the Ansatz.

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
            blockwise_parameter: The base parameters to be used. Can be used to couple the
                parameters of certain blocks.
            initial_state: An `InitialState` object to prepent to the Ansatz.
                TODO deprecate this feature in favour of prepend or overloading __add__ in
                the initial state class

        Raises:
            TypeError: If `blocks` contains an unsupported object.
            ValueError: If the initial state has less qubits than specified via the blocks or
                qubit indices.
            ValueError: If the ``overwrite_block_parameters`` is set to a list of list of
                Parameters but does not match the total number of blocks in the final circuit.

        Examples:
            TODO
        """
        # insert barriers in between the blocks?
        self._insert_barriers = insert_barriers

        # get blocks in the right format
        if blocks is None:
            blocks = []

        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        # convert all input blocks to the same type
        self._blocks = []
        for block in blocks:
            self._blocks += [self._convert_to_block(block)]

        # get reps in the right format
        if reps is None:  # if reps is None, set it to [0, ..., len(num_blocks) - 1]
            self._reps = 1
            self._replist = list(range(len(self._blocks)))
        elif isinstance(reps, int):  # if reps is an int, set it to reps * [0, ..., len(blocks) - 1]
            self._reps = reps
            self._replist = reps * list(range(len(self._blocks)))
        else:
            self._reps = None
            self._replist = reps

        # get qubit_indices in the right format (i.e. list of lists)
        if qubit_indices is None:  # if None, set the indices to [0, ..., block.num_qubits - 1]
            self._qargs = [list(range(block.num_qubits)) for block in self._blocks]
        elif not isinstance(qubit_indices[0], list):  # user provided a flat, single list
            self._qargs = [qubit_indices]
        else:  # right format
            self._qargs = qubit_indices

        # get the maximum number of qubits from the qubit indices
        self._num_qubits = int(numpy.max(self._qargs) + 1 if len(self._qargs) > 0 else 0)

        # If there is an initial state object, check that the number of qubits is compatible
        # construct the circuit immediately. If the InitialState could modify the number of qubits
        # we could also do this later at circuit construction.
        self._initial_state = initial_state
        self._initial_state_circuit = None
        if initial_state is not None:
            # construct the circuit of the initial state
            self._initial_state_circuit = initial_state.construct_circuit(mode='circuit')

            # the initial state dictates the number of qubits since we do not have information
            # about on which qubits the initial state acts
            if self._initial_state_circuit.n_qubits < self._num_qubits:
                raise ValueError('The provided initial state has less qubits than the Ansatz.')

            self._num_qubits = self._initial_state_circuit.n_qubits

        # keep track of the circuit
        self._circuit = None

        # Set up the base and surface parameters:
        # The surface parameters are the parameters the user has access to. The base parameters
        # are the parameters used in the internally stored circuit. Keeping two separate lists
        # of parameters allows us to bind and substitute values as the user specifies without
        # loosing track of which parameters exist.
        if blockwise_parameters is None:
            self._surface_params = []  # the parameters the user can access
            self._base_params = []  # the internally used parameters
            self._blockwise_base_params = []  # per block, used for convenience later on
            # fill the parameter lists
            param_count = 0
            for idx in self._replist:
                block_params = get_parameters(self._blocks[idx])  # get the parameters per block
                self._surface_params += block_params  # add them to the surface parameters

                # set the base parameters per block
                n = len(block_params)
                block_base_params = [
                    Parameter('θ{}'.format(i)) for i in range(param_count, param_count + n)
                ]
                param_count += n
                self._blockwise_base_params += [block_base_params]
                self._base_params += block_base_params
        else:
            print(blockwise_parameters)
            if len(blockwise_parameters) != len(self._replist):
                raise ValueError('Number of blockwise parameters does not fit the number of blocks')
            self._blockwise_base_params = blockwise_parameters
            all_parameters = []
            for block_parameters in blockwise_parameters:
                all_parameters += block_parameters
            self._base_params = list(set(all_parameters))
            self._surface_params = list(set(all_parameters))

        # parameter bounds
        self._bounds = None

    def _convert_to_block(self, layer: Any) -> Instruction:
        """Try to convert `layer` to an Instruction.

        Args:
            layer: The object to be converted to an Ansatz block / Instruction.

        Returns:
            The layer converted to an Instruction.

        Raises:
            TypeError: If the input cannot be converted to an Instruction.
        """
        if isinstance(layer, Instruction):
            return layer
        elif hasattr(layer, 'to_instruction'):
            return layer.to_instruction()
        else:
            raise TypeError('Adding a {} to an Ansatz is not supported.'.format(type(layer)))

    @property
    def setting(self):
        """TODO Deprecate.

        Returns information about the setting.
        """
        ret = "variational form: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @property
    def preferred_init_points(self):
        """TODO Deprecate.

        Returns preferred init points."""
        return None

    @property
    def support_parameterized_circuit(self):
        """TODO Deprecate.

        Whether it is supported to bind parameters in this circuit.
        """
        return True

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this Ansatz.

        Returns:
            The number of qubits.
        """
        return int(self._num_qubits)

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds.

        TODO change to return (-np.inf, np.inf) as unbounded?

        Returns:
            A list of pairs indicating the bounds, as (lower, upper).
            None indicates an unbounded parameter in the corresponding direction.
            If None is returned, problem is fully unbounded.
        """
        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def params(self) -> Union[List[float], List[Parameter]]:
        """Get the parameters of the Ansatz.

        Only the so-called "surface parameters" of the Ansatz are subject to change, these
        can be modified and re-assigned to new values. Below, the Ansatz keeps track of the
        "base parameters" which remain unique.

        Returns:
            A list containing the surface parameters.
        """
        return self._surface_params

    @params.setter
    def params(self, params: Union[dict, List[float], List[Parameter], ParameterVector]) -> None:
        """Set the parameters of the Ansatz.

        This sets the surface parameters, not the base parameters.

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
            self._surface_params = new_params

        # if a list is provided, just assign if the sizes match
        else:
            if len(params) != self.num_parameters:
                raise ValueError('Mismatching number of parameters! '
                                 'Provided: {}, required: {}'
                                 ''.format(len(params), self.num_parameters))
            self._surface_params = params

    def bind_parameters(self, params: Union[List[float], List[Parameter], ParameterVector]

                        ) -> QuantumCircuit:
        """Bind ``params`` to the underlying circuit of the Ansatz.

        This method allows handling of both ``qiskit.circuit.Parameter`` objects and numbers.
        It returns a copy of the internally stored circuit with the new specified parameters.

        Returns:
            A copy of the Ansatz circuit with the specified parameters.

        Raises:
            TypeError: If ``params`` contains an unsupported type.
        """
        if all(isinstance(param, numbers.Real) for param in params):
            param_dict = dict(zip(self._base_params, params))
            circuit_copy = self._circuit.bind_parameters(param_dict)

        # if they are new parameters, replace them in the circuit
        elif all(isinstance(param, Parameter) for param in params):
            param_dict = dict(zip(self._base_params, params))
            circuit_copy = self._circuit.copy()
            circuit_copy._substitute_parameters(param_dict)

        # otherwise the input type is not supported
        else:
            raise TypeError('Unsupported type of `params`, {}'.format(type(params)))

        return circuit_copy

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in the Ansatz.

        Returns:
            The number of parameters.
        """
        return len(self._base_params)  # could also be len(self._surface_params)

    def construct_circuit(self,
                          parameters: Union[List[float], List[Parameter], ParameterVector],
                          q: Optional[QuantumRegister] = None,
                          ) -> QuantumCircuit:
        """Deprecated, use `to_circuit()`.

        Args:
            parameters: The parameters for the Ansatz.
            q: The qubit register to use to build the circuit. If None, a new register with the
                name 'q' is created.

        Returns:
            The Ansatz as circuit with the specified parameters.

        Raises:
            ValueError: If the qubit register is provided but the length does not coincide with the
                number of qubits of the Ansatz.
        """
        self.params = parameters
        if q is None:
            circuit = QuantumCircuit(self.num_qubits)
        elif len(q) != self.num_qubits:
            raise ValueError('The size of the register is not equal to the number of qubits.')
        else:
            circuit = QuantumCircuit(q)

        circuit += self.to_circuit()
        return circuit

    def _parametrize_block(self, block: Instruction, qargs: List[int],
                           params: Optional[List[Parameter]]) -> QuantumCircuit:
        """Convert ``block`` to a circuit of correct width and parameterized with the
        specified parameters.

        Args:
            block: The instruction to which the base parameters are bound.
            qargs: The indices for the block.
            params: The parameters to bind to the block.

        Returns:
            The block as circuit of width ``self.num_qubits`` where the parameters have been
            substituted as specified.
        """
        circuit = QuantumCircuit(self.num_qubits)
        circuit.append(block, qargs)
        if params is not None:
            update = dict(zip(list(circuit.parameters), params))
            circuit = circuit.copy()
            circuit._substitute_parameters(update)

        return circuit

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
                if self._initial_state:
                    circuit = self._initial_state.construct_circuit(mode='circuit')
                else:
                    circuit = QuantumCircuit(self.num_qubits)

                # add the blocks, if they are specified
                if len(self._replist) > 0:
                    # the first block (separately so barriers can be inserted in the for-loop)
                    param_idx, idx = 0, self._replist[0]
                    block, qargs = self._blocks[idx], self._qargs[idx]
                    params = self._blockwise_base_params[param_idx]
                    circuit.extend(self._parametrize_block(block, qargs, params))

                    for param_idx, idx in enumerate(self._replist[1:]):
                        if self._insert_barriers:
                            circuit.barrier()
                        block, qargs = self._blocks[idx], self._qargs[idx]
                        params = self._blockwise_base_params[param_idx + 1]
                        circuit.extend(self._parametrize_block(block, qargs, params))

            # store the circuit
            self._circuit = circuit

        # TODO make this on parameter change only?
        return self.bind_parameters(self._surface_params)

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
                       'rxx', 'ryy', 'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap', 'cswap',
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

        # keep track of which blocks to add to the Ansatz
        self._replist += [len(self._blocks) - 1]

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

        # update the parameters
        new_base_params = [Parameter('θ{}'.format(self.num_parameters + i))
                           for i in range(len(get_parameters(block)))]
        self._blockwise_base_params += [new_base_params]
        self._base_params += new_base_params
        self._surface_params += get_parameters(block)

        # modify the circuit accordingly
        if self._circuit is None:
            _ = self.to_circuit()  # automatically constructed
        else:
            if self._insert_barriers and len(self._replist) > 1:
                self._circuit.barrier()

            block, qargs = self._blocks[-1], self._qargs[-1]
            params = self._blockwise_base_params[-1]
            self._circuit.extend(self._parametrize_block(block, qargs, params))

        return self

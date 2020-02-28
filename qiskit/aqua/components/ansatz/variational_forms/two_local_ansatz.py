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

"""The two-local gate Ansatz.

TODO
    * remove the temporary param subst fix and move to ccts away from gates
    * if entanglement is not a callable, store only 2 blocks, not all of them
    * let identify gate return a type if possible to avoid substitution, handle the circuit
        case differently
"""

from typing import Union, Optional, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.extensions.standard import (IGate, XGate, YGate, ZGate, HGate, TGate, SGate, TdgGate,
                                        SdgGate, RXGate, RXXGate, RYGate, RYYGate, RZGate, SwapGate,
                                        CXGate, CYGate, CZGate, CHGate, CRXGate, CRYGate, CRZGate)

from qiskit.aqua import AquaError
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map
from qiskit.aqua.components.initial_states import InitialState

from qiskit.aqua.components.ansatz import Ansatz


class TwoLocalAnsatz(Ansatz):
    """The two-local gate Ansatz.

    TODO
    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 reps: int = 3,
                 rotation_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement: Union[str, List[List[int]], callable] = 'full',
                 initial_state: Optional[InitialState] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            reps: Specifies how often a block of consisting of a rotation layer and entanglement
                layer is repeated.
            rotation_gates: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. 'ry') or the gate type itself (e.g. RYGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_gates: The gates used in the entanglement layer. Can be specified in
                the same format as `rotation_gates`.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            initial_state: An `InitialState` object to prepent to the Ansatz.
                TODO deprecate this feature in favour of prepend or overloading __add__ in
                the initial state class
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of `qiskit.circuit.Parameter`. The name of each parameter is the
                number of its occurrence with this specified prefix.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
                Defaults to False.

        Examples:
            >>> ansatz = TwoLocalAnsatz(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the Ansatz
            >>> qc += ansatz.to_circuit()
            >>> qc.decompose().draw()  # decompose the layers into standard gates
                    ┌────────┐ ░            ░ ┌────────┐ ░            ░ ┌────────┐
            q_0: |0>┤ Ry(_0) ├─░───■────────░─┤ Ry(_3) ├─░───■────────░─┤ Ry(_6) ├
                    ├────────┤ ░ ┌─┴─┐      ░ ├────────┤ ░ ┌─┴─┐      ░ ├────────┤
            q_1: |0>┤ Ry(_1) ├─░─┤ X ├──■───░─┤ Ry(_4) ├─░─┤ X ├──■───░─┤ Ry(_7) ├
                    ├────────┤ ░ └───┘┌─┴─┐ ░ ├────────┤ ░ └───┘┌─┴─┐ ░ ├────────┤
            q_2: |0>┤ Ry(_2) ├─░──────┤ X ├─░─┤ Ry(_5) ├─░──────┤ X ├─░─┤ Ry(_8) ├
                    └────────┘ ░      └───┘ ░ └────────┘ ░      └───┘ ░ └────────┘

            >>> ansatz = TwoLocalAnsatz(3, ['ry', 'rz'], 'cz', 'full', reps=1, insert_barriers=True)
            >>> print(ansatz)  # quick way of plotting the Ansatz
                    ┌────────┐┌────────┐ ░           ░  ┌────────┐ ┌────────┐
            q_0: |0>┤ Ry(_0) ├┤ Rz(_1) ├─░──■──■─────░──┤ Ry(_6) ├─┤ Rz(_7) ├
                    ├────────┤├────────┤ ░  │  │     ░  ├────────┤ ├────────┤
            q_1: |0>┤ Ry(_2) ├┤ Rz(_3) ├─░──■──┼──■──░──┤ Ry(_8) ├─┤ Rz(_9) ├
                    ├────────┤├────────┤ ░     │  │  ░ ┌┴────────┤┌┴────────┤
            q_2: |0>┤ Ry(_4) ├┤ Rz(_5) ├─░─────■──■──░─┤ Ry(_10) ├┤ Rz(_11) ├
                    └────────┘└────────┘ ░           ░ └─────────┘└─────────┘

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ansatz = TwoLocalAnsatz(3, 'x', 'crx', entangler_map, reps=1)
            >>> print(ansatz)  # note: no barriers inserted this time!
                    ┌───┐                         ┌────────┐┌───┐
            q_0: |0>┤ X ├────■────────────────────┤ Rx(_2) ├┤ X ├
                    ├───┤┌───┴────┐          ┌───┐└───┬────┘└───┘
            q_1: |0>┤ X ├┤ Rx(_0) ├────■─────┤ X ├────┼──────────
                    ├───┤└────────┘┌───┴────┐└───┘    │     ┌───┐
            q_2: |0>┤ X ├──────────┤ Rx(_1) ├─────────■─────┤ X ├
                    └───┘          └────────┘               └───┘

            >>> entangler_map = [[0, 3], [0, 2]]  # entangle the first and last two-way
            >>> ansatz = TwoLocalAnsatz(4, [], 'cry', entangler_map, reps=1)
            >>> circuit = ansatz.to_circuit() + ansatz.to_circuit()  # add two Ansaetze
            >>> circuit.decompose().draw()  # note, that the parameters are the same!
            q_0: |0>────■─────────■─────────■─────────■─────
                        │         │         │         │
            q_1: |0>────┼─────────┼─────────┼─────────┼─────
                        │     ┌───┴────┐    │     ┌───┴────┐
            q_2: |0>────┼─────┤ Ry(_1) ├────┼─────┤ Ry(_1) ├
                    ┌───┴────┐└────────┘┌───┴────┐└────────┘
            q_3: |0>┤ Ry(_0) ├──────────┤ Ry(_0) ├──────────
                    └────────┘          └────────┘
        """
        # initialize Ansatz
        super().__init__(insert_barriers=insert_barriers, initial_state=initial_state)

        # store arguments needing no pre-processing
        self._num_qubits = num_qubits
        self._entanglement = entanglement
        self._parameter_prefix = parameter_prefix
        self._param_count = 0
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._skip_final_rotation_layer = skip_final_rotation_layer

        # handle the single- and two-qubit gate specifications
        self.rotation_gates = rotation_gates or []
        self.entanglement_gates = entanglement_gates or []

    def _get_new_parameters(self, n):
        new_parameters = [Parameter('{}{}'.format(self._parameter_prefix, i + self._param_count))
                          for i in range(n)]
        self._param_count += n
        return new_parameters

    def _get_rotation_layer(self, block_num: int) -> Gate:
        """Get the rotation layer for the current block.

        Args:
            block_num: The index of the current block.

        Returns:
            The rotation layer as Gate.
        """
        # determine the entangled qubits for this block
        if self._skip_unentangled_qubits:
            all_qubits = []
            for src, tgt in self.get_entangler_map(block_num):
                all_qubits.extend([src, tgt])
            entangled_qubits = sorted(list(set(all_qubits)))
        else:
            entangled_qubits = list(range(self.num_qubits))

        # build the circuit for this block
        circuit = QuantumCircuit(self.num_qubits, name='rot{}'.format(block_num))

        # iterate over all qubits
        for qubit in range(self.num_qubits):

            # check if we need to apply the gate to the qubit
            if not self._skip_unentangled_qubits or qubit in entangled_qubits:

                # apply the gates
                for gate, num_params in self.rotation_gates:
                    if num_params == 0:
                        circuit.append(gate, [qubit], [])  # todo: use _append with register
                    else:
                        # define the new parameters, named '_{}', where {} gets
                        # replaced by the number of the parameter
                        # the list here is needed since a gate might take
                        # more than one parameter (e.g. a general rotation)
                        # param_count = self.num_parameters + len(circuit.parameters)
                        # params = [Parameter('{}{}'.format(self._parameter_prefix, param_count + i))
                                #   for i in range(num_params)]
                        # params = [Parameter('{}'.format(param_count + i))
                        #           for i in range(num_params)]
                        # param_count += num_params
                        params = self._get_new_parameters(num_params)

                        # correctly replace the parameters
                        sub_circuit = QuantumCircuit(self.num_qubits)
                        sub_circuit.append(gate, [qubit], [])
                        update = dict(zip(list(sub_circuit.parameters), params))
                        sub_circuit._substitute_parameters(update)

                        # add the gate
                        circuit.extend(sub_circuit)

        print('===========')
        print('returning')
        print(circuit)
        print('===========')
        return circuit.to_gate()

    def _get_entanglement_layer(self, block_num: int) -> Gate:
        """Get the entangler map for this block.

        For some kinds of entanglement (e.g. 'sca') the entangler map is differs in different
        blocks, therefore we query the entangler map every time. For constant schemata, such as
        'linear', this is slightly inefficient, since the entangler map does not change.
        However, the number of times get_entangler_map is called equals to `reps` which usually is
        of O(10), and therefore most likely no bottleneck

        Args:
            block_num: The index of the current block.

        Returns:
            The entanglement layer as gate.
        """

        circuit = QuantumCircuit(self.num_qubits, name='ent{}'.format(block_num))

        for src, tgt in self.get_entangler_map(block_num):
            # apply the gates
            for gate, num_params in self.entanglement_gates:
                if num_params == 0:
                    circuit.append(gate, [src, tgt], [])
                else:
                    # param_count = self.num_parameters + len(circuit.parameters)
                    # params = [Parameter('{}{}'.format(self._parameter_prefix, param_count + i))
                    #           for i in range(num_params)]
                    # params = [Parameter('{}'.format(param_count + i)) for i in range(num_params)]
                    # param_count += num_params
                    params = self._get_new_parameters(num_params)

                    # correctly replace the parameters
                    sub_circuit = QuantumCircuit(self.num_qubits)
                    sub_circuit.append(gate, [src, tgt], [])
                    update = dict(zip(list(sub_circuit.parameters), params))
                    sub_circuit._substitute_parameters(update)

                    # add the gate
                    circuit.extend(sub_circuit)

        return circuit.to_gate()

    @property
    def rotation_gates(self) -> List[Tuple[type, int]]:
        """Return a the single qubit gate (or gates) in tuples of callable and number of parameters.

        The reason this is implemented as separate function is that the user can set up a class
        with special single and two qubit gates, for cases we do not cover in identify gate.
        And this design "outsources" the identification of the gate from the main code that
        builds the circuit, which makes the code more modular.

        Returns:
            list[tuple]: the single qubit gate(s) as tuples (QuantumCircuit.gate, num_parameters),
                e.g. (QuantumCircuit.x, 0) or (QuantumCircuit.ry, 1)
        """
        gate_param_list = [TwoLocalAnsatz.identify_gate(gate) for gate in self._rotation_gates]
        return gate_param_list

    @rotation_gates.setter
    def rotation_gates(self, gates):
        """Set new rotation gates."""
        # invalidate circuit definition
        self._circuit = None

        if not isinstance(gates, list):
            self._rotation_gates = [gates]
        else:
            self._rotation_gates = gates

    @property
    def entanglement_gates(self) -> List[Tuple[type, int]]:
        """
        Return a the twos qubit gate(or gates) in form of callable(s).

        Returns:
            list[tuple]: the single qubit gate(s) as tuples (QuantumCircuit.gate, num_parameters),
                e.g. (QuantumCircuit.cx, 0) or (QuantumCircuit.cry, 1)
        """
        gate_param_list = [TwoLocalAnsatz.identify_gate(gate) for gate in self._entanglement_gates]
        return gate_param_list

    @entanglement_gates.setter
    def entanglement_gates(self, gates):
        """Set new entanglement gates."""
        # invalidate circuit definition
        self._circuit = None

        if not isinstance(gates, list):
            self._entanglement_gates = [gates]
        else:
            self._entanglement_gates = gates

    # @property
    # def parameters(self):
        # """Return the parameters."""
        # return [Parameter('t{i}') for i in range(self.num_parameters)]

    @staticmethod
    def identify_gate(gate: Union[str, type, QuantumCircuit]) -> Tuple[type, int]:
        """For a gate provided as str (e.g. 'ry') or type (e.g. RYGate) this function returns the
        according gate type along with the number of parameters (e.g. (RYGate, 1)).

        Args:
            gate: The qubit gate.

        Returns:
            The specified gate with the required number of parameters.

        Raises:
            AquaError: The type of `gate` is invalid.
            AquaError: The type of `gate` is str but the name is unknown.
            AquaError: The type of `gate` is type but the gate type is unknown.

        Note:
            Outlook: If gates knew their number of parameters as static property, we could also
            allow custom gate types.
        """
        if isinstance(gate, QuantumCircuit):
            return (gate.to_gate(), len(gate.parameters))

        # check the list of valid gates
        # this could be a lot easier if the standard gates would have `name` and `num_params`
        # as static types, which might be something they should have anyways
        theta = Parameter('θ')
        valid_gates = {
            'ch': (CHGate(), 0),
            'cx': (CXGate(), 0),
            'cy': (CYGate(), 0),
            'cz': (CZGate(), 0),
            'crx': (CRXGate(theta), 1),
            'cry': (CRYGate(theta), 1),
            'crz': (CRZGate(theta), 1),
            'h': (HGate(), 0),
            'i': (IGate(), 0),
            'id': (IGate(), 0),
            'iden': (IGate(), 0),
            'rx': (RXGate(theta), 1),
            'rxx': (RXXGate(theta), 1),
            'ry': (RYGate(theta), 1),
            'ryy': (RYYGate(theta), 1),
            'rz': (RZGate(theta), 1),
            's': (SGate(), 0),
            'sdg': (SdgGate(), 0),
            'swap': (SwapGate(), 0),
            'x': (XGate(), 0),
            'y': (YGate(), 0),
            'z': (ZGate(), 0),
            't': (TGate(), 0),
            'tdg': (TdgGate(), 0),
        }

        if isinstance(gate, str):
            # iterate over the gate names and look for the specified gate
            for identifier, (standard_gate, num_params) in valid_gates.items():
                if gate == identifier:
                    return (standard_gate, num_params)
            raise AquaError('Unknown gate name `{}`.'.format(gate))

        if isinstance(gate, type):
            # iterate over the gate types and look for the specified gate
            for _, (standard_gate, num_params) in valid_gates.items():
                if isinstance(standard_gate, gate):
                    return (standard_gate, num_params)
            raise AquaError('Unknown gate type`{}`.'.format(gate))

        raise AquaError('Invalid input type {}. '.format(type(gate))
                        + '`gate` must be a type, str or QuantumCircuit.')

    def get_entangler_map(self, offset: int = 0) -> List[List[int]]:
        """Return the specified entangler map, if self._entangler_map if it has been set previously.

        Args:
            offset (int): Some entanglements allow an offset argument, since the entangler map might
                differ per entanglement block (e.g. for 'sca' entanglement). This is the block
                index.

        Returns:
            A list of [src, tgt] pairs specifying entanglements, also known as entangler map.

        Raises:
            AquaError: Unsupported format of entanglement, if self._entanglement has the wrong
                format.
        """
        if isinstance(self._entanglement, str):
            return get_entangler_map(self._entanglement, self.num_qubits, offset)
        elif callable(self._entanglement):
            return validate_entangler_map(self._entanglement(offset), self.num_qubits)
        elif isinstance(self._entanglement, list):
            return validate_entangler_map(self._entanglement, self.num_qubits)
        else:
            raise AquaError('Unsupported format of entanglement!')

    def _set_blocks(self):
        """Set the blocks according to the current state."""
        if self.num_qubits is None:
            raise AquaError('The number of qubits has not been set!')

        if self.rotation_gates is None:
            raise AquaError('No rotation gates are specified.')

        if self.entanglement_gates is None:
            raise AquaError('No entanglement gates are specified.')

        blocks = []
        self._param_count = 0
        # define the blocks of this Ansatz
        for block_num in range(self._reps):
            # append a rotation layer, if entanglement gates are specified
            if len(self._rotation_gates) > 0:
                blocks += [self._get_rotation_layer(block_num)]

            # append an entanglement layer, if entanglement gates are specified
            if len(self._entanglement_gates) > 0:
                blocks += [self._get_entanglement_layer(block_num)]

        # add a final rotation layer, if not specified otherwise
        if not self._skip_final_rotation_layer and len(self._rotation_gates) > 0:
            blocks += [self._get_rotation_layer(self._reps)]

        # TODO: allow adding a bunch of blocks at once
        self._overwrite_block_parameters = False
        self.blocks = blocks

    def to_circuit(self):
        """Construct the circuit."""
        self._set_blocks()
        return super().to_circuit()

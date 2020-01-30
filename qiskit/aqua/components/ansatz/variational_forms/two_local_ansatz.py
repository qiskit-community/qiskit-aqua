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
"""

from typing import Union, Optional, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.extensions.standard import (IdGate, XGate, YGate, ZGate, HGate, TGate, SGate, TdgGate,
                                        SdgGate, RXGate, RYGate, RZGate, SwapGate, Barrier,
                                        CnotGate, CyGate, CzGate, CHGate, CrxGate, CryGate, CrzGate)

from qiskit.aqua import AquaError
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map

from qiskit.aqua.components.ansatz import Ansatz


class TwoLocalAnsatz(Ansatz):
    """The two-local gate Ansatz.

    TODO
    """

    def __init__(self,
                 num_qubits: int,
                 rotation_gates: Union[str, List[str], type, List[type]],
                 entanglement_gates: Union[str, List[str], type, List[type]],
                 entanglement: Union[str, List[List[int]], callable] = 'full',
                 reps: Optional[int] = 3,
                 parameter_prefix: str = '_',
                 insert_barriers: bool = False,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = True) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
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
            reps: Specifies how often a block of consisting of a rotation layer and entanglement
                layer is repeated.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of `qiskit.circuit.Parameter`. The name of each parameter is the
                number of its occurrence with this specified prefix.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
                Defaults to False.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.

        Examples:
            >>> ansatz = TwoLocalAnsatz(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the Ansatz
            >>> qc += ansatz.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> ansatz = TwoLocalAnsatz(3, ['ry', 'rz'], 'cz', 'full', reps=2, insert_barriers=True)
            >>> ansatz.to_circuit().draw()  # quick way of plotting the Ansatz
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ansatz = TwoLocalAnsatz(3, 'x', 'crx', entangler_map, reps=2)
            >>> ansatz.to_circuit().draw()
            TODO: circuit diagram

            >>> entangler_map = [[0, 3], [3, 0]]  # entangle the first and last two-way
            >>> ansatz = TwoLocalAnsatz(4, [], 'cry', entangler_map, reps=2)
            >>> circuit = ansatz.to_circuit() + ansatz.to_circuit()  # add two Ansaetze
            >>> circuit.draw()
            TODO: circuit diagram

            >>> ansatz = TwoLocalAnsatz(3, 'ry', 'cx', 'linear', final_rotation_layer=False)
            >>> ansatz.to_circuit().draw()
            TODO: circuit diagram
        """
        # initialize Ansatz
        super().__init__(num_qubits, insert_barriers=insert_barriers)

        # store arguments needing no pre-processing
        self._entanglement = entanglement
        self._parameter_prefix = parameter_prefix
        self._skip_unentangled_qubits = skip_unentangled_qubits

        # handle the single- and two-qubit gate specifications
        if rotation_gates is None:
            self._rotation_gates = []
        elif not isinstance(rotation_gates, list):
            self._rotation_gates = [rotation_gates]
        else:
            self._rotation_gates = rotation_gates

        if entanglement_gates is None:
            self._entanglement_gates = []
        elif not isinstance(entanglement_gates, list):
            self._entanglement_gates = [entanglement_gates]
        else:
            self._entanglement_gates = entanglement_gates

        for block_num in range(reps):
            if insert_barriers and block_num > 0:
                self.append(Barrier(self.num_qubits))
            self.append(self._get_rotation_layer(block_num))

            if insert_barriers:
                self.append(Barrier(self.num_qubits))
            self.append(self._get_entanglement_layer(block_num))

        if not skip_final_rotation_layer:
            if insert_barriers and reps > 0:
                self.append(Barrier(self.num_qubits))
            self.append(self._get_rotation_layer(reps))

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
        circuit = QuantumCircuit(self.num_qubits)

        # iterate over all qubits
        for qubit in range(self.num_qubits):

            # check if we need to apply the gate to the qubit
            if not self._skip_unentangled_qubits or qubit in entangled_qubits:

                # apply the gates
                for gate, num_params in self.single_qubit_gate:
                    if num_params == 0:
                        circuit.append(gate(), [qubit], [])
                    else:
                        # define the new parameters, named '_{}', where {} gets
                        # replaced by the number of the parameter
                        # the list here is needed since a gate might take
                        # more than one parameter (e.g. a general rotation)
                        param_count = self.num_parameters + len(circuit.parameters)
                        params = [Parameter('{}{}'.format(self._parameter_prefix, param_count + i))
                                  for i in range(num_params)]

                        # add the gate
                        circuit.append(gate(*params), [qubit], [])

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

        circuit = QuantumCircuit(self.num_qubits)

        for src, tgt in self.get_entangler_map(block_num):
            # apply the gates
            for gate, num_params in self.two_qubit_gate:
                if num_params == 0:
                    circuit.append(gate(), [src, tgt], [])
                else:
                    param_count = self.num_parameters + len(circuit.parameters)
                    params = [Parameter('{}{}'.format(self._parameter_prefix, param_count + i))
                              for i in range(num_params)]

                    # add the gate
                    circuit.append(gate(*params), [src, tgt], [])

        return circuit.to_gate()

    @property
    def rotation_gates(self):
        """
        Return a the single qubit gate (or gates) in tuples of callable and number of parameters.

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

    @property
    def entanglement_gates(self):
        """
        Return a the twos qubit gate(or gates) in form of callable(s).

        Returns:
            list[tuple]: the single qubit gate(s) as tuples (QuantumCircuit.gate, num_parameters),
                e.g. (QuantumCircuit.cx, 0) or (QuantumCircuit.cry, 1)
        """
        gate_param_list = [TwoLocalAnsatz.identify_gate(gate) for gate in self._entanglement_gates]
        return gate_param_list

    @staticmethod
    def identify_gate(gate: Union[str, type]) -> Tuple[type, int]:
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
        # check the list of valid gates
        # this could be a lot easier if the standard gates would have `name` and `num_params`
        # as static types, which might be something they should have anyways
        valid_gates = {
            'iden': (IdGate, 0),
            'x': (XGate, 0),
            'y': (YGate, 0),
            'z': (ZGate, 0),
            'h': (HGate, 0),
            't': (TGate, 0),
            'tdg': (TdgGate, 0),
            's': (SGate, 0),
            'sdg': (SdgGate, 0),
            'rx': (RXGate, 1),
            'ry': (RYGate, 1),
            'rz': (RZGate, 1),
            'swap': (SwapGate, 0),
            'cx': (CnotGate, 0),
            'cy': (CyGate, 0),
            'cz': (CzGate, 0),
            'ch': (CHGate, 0),
            'crx': (CrxGate, 1),
            'cry': (CryGate, 1),
            'crz': (CrzGate, 1),
        }

        if isinstance(gate, str):  # pylint-disable: no-else-raise
            # iterate over the gate names and look for the specified gate
            for identifier, (gate_type, num_params) in valid_gates.items():
                if gate == identifier:
                    return (gate_type, num_params)
            raise AquaError('Unknown gate name `{}`.'.format(gate))

        elif isinstance(gate, type):
            # iterate over the gate types and look for the specified gate
            for _, (gate_type, num_params) in valid_gates.items():
                if gate == gate_type:
                    return (gate_type, num_params)
            raise AquaError('Unknown gate type`{}`.'.format(gate))

        else:
            raise AquaError('Invalid input, `gate` must be a str.')

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

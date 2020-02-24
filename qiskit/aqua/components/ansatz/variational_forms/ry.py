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

"""The RY variational form."""

from typing import Union, Optional, List, Tuple

import numpy as np
from qiskit.extensions.standard import RYGate, CZGate
from qiskit.aqua.components.initial_states import InitialState
from .two_local_ansatz import TwoLocalAnsatz


class RY(TwoLocalAnsatz):
    """The RY variational form.

    TODO
    """

    def __init__(self,
                 num_qubits: int,
                 reps: Optional[int] = 3,
                 entanglement_gates: Union[str, List[str], type, List[type]] = CZGate,
                 entanglement: Union[str, List[List[int]], callable] = 'full',
                 initial_state: Optional[InitialState] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = '_',
                 insert_barriers: bool = False
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            reps: Specifies how often a block of consisting of a rotation layer and entanglement
                layer is repeated.
            entanglement_gates: The gates used in the entanglement layer. Can be specified via the
                name of a gate (e.g. 'cx') or the gate type itself (e.g. CnotGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
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
            >>> ry = RY(3)  # create the variational form on 3 qubits
            >>> print(ry)  # show the circuit
            TODO: circuit diagram

            >>> ry = RY(3, entanglement='linear', reps=2, insert_barriers=True)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
            >>> qc += ry.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> rycrx = RY(2, entanglement_gate='crx', 'sca', reps=2, insert_barriers=True)
            >>> print(rycrx)
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ry = RY(3, 'cx', entangler_map, reps=2)
            >>> print(ry)
            TODO: circuit diagram

            >>> ry_linear = RY(2, entanglement='linear', reps=1)
            >>> ry_full = RY(2, entanglement='full', reps=1)
            >>> my_ry = ry_linear + ry_full
            >>> print(my_ry)
        """
        super().__init__(num_qubits,
                         reps=reps,
                         rotation_gates=RYGate,
                         entanglement_gates=entanglement_gates,
                         entanglement=entanglement,
                         initial_state=initial_state,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers)

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-np.pi, np.pi)]

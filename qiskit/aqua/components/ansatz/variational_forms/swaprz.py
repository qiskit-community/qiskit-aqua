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

"""The SwapRZ variational form.

TODO
* implement  the RYY Gate
* test if this actually coincides with the current SwapRZ varform
"""

from typing import Union, Optional, List, Tuple

from qiskit.extensions.standard import RZGate, RXXGate, RYYGate
from .two_local_ansatz import TwoLocalAnsatz


class SwapRZ(TwoLocalAnsatz):
    r"""The SwapRZ variational form.

    The SwapRZ variational form consists of layers of Z-rotations and (XX + YY)-rotations as
    entanglements. These rotations can be written as

    .. math::

        R_Z(\theta) = e^{-i \theta Z}

    and

    .. math::

        R_{XX+YY}(\theta) = e^{-i \theta (X \otimes X + Y \otimes Y)}
                          \approx e^{-i \theta X \otimes X} e^{-i \theta Y \otimes Y }
                          = R_{XX}(\theta) R_{YY}(\theta)

    where the approximation used comes from the Trotter expansion of the sum in the exponential.

    This variational form is used for TODO
    """

    def __init__(self,
                 num_qubits: int,
                 entanglement_gates: Union[str, List[str], type, List[type]] = CzGate,
                 entanglement: Union[str, List[List[int]], callable] = 'full',
                 reps: Optional[int] = 3,
                 parameter_prefix: str = '_',
                 insert_barriers: bool = False,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
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
            >>> ry = RYRZ(3)  # create the variational form on 3 qubits
            >>> print(ryrz)  # show the circuit
            TODO: circuit diagram

            >>> ryrz = RYRZ(4, entanglement='full', reps=1)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
            >>> qc += ryrz.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> ryrz_crx = RYRZ(2, entanglement_gate='crx', 'sca', reps=1, insert_barriers=True)
            >>> print(ryrz_crx)
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ry = RYRZ(3, 'cx', entangler_map, reps=2)
            >>> print(ryrz)
            TODO: circuit diagram

            >>> ryrz = RYRZ(2, entanglement='linear', reps=1)
            >>> ry = RY(2, entanglement='full', reps=1)
            >>> my_varform = ryrz + ry
            >>> print(my_varform)
        """
        super().__init__(num_qubits,
                         rotation_gates=RZGate,
                         entanglement_gates=[RXXGate, RYYGate],
                         entanglement=entanglement,
                         reps=reps,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         skip_final_rotation_layer=skip_final_rotation_layer)

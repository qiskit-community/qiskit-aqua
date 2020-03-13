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

"""The SwapRZ variational form."""

from typing import Union, Optional, List, Tuple, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.extensions.standard import RZGate
from qiskit.util import deprecate_arguments
from qiskit.aqua.components.initial_states import InitialState

from .two_local_ansatz import TwoLocalAnsatz


class SwapRZ(TwoLocalAnsatz):
    r"""The SwapRZ ansatz.

    This trial wave function is layers of swap plus :math:`z` rotations with entanglements.
    It was designed principally to be a particle-preserving variational form for
    :mod:`qiskit.chemistry`. Given an initial state as a set of 1's and 0's it will preserve
    the number of 1's - where for chemistry a 1 will indicate a particle.

    Note:

        In chemistry, to define the particles for SwapRZ, use a
        :class:`~qiskit.chemistry.components.initial_states.HartreeFock` initial state with
        the `Jordan-Wigner` qubit mapping

    For the case of none of qubits are unentangled to other qubits, the number of optimizer
    parameters `SwapRZ` creates and uses is given by
    :math:`q + d \times \left(q + \sum_{k=0}^{q-1}|D(k)|\right)`, where :math:`|D(k)|` denotes the
    *cardinality* of :math:`D(k)` or, more precisely, the *length* of :math:`D(k)`
    (since :math:`D(k)` is not just a set, but a list).
    Nonetheless, in some cases the entanglement does not include all qubits, that is, some
    qubits are not entangled by other qubits. Then the number of parameters is reduced by
    :math:`d \times q'`, where :math:`q'` is the number of unentangled qubits.
    This is because adding more Rz gates to the unentangled qubits only introduce overhead without
    bringing any benefit; furthermore, theoretically, applying multiple Rz gates in a row can be
    reduced to a single Rz gate with the summed rotation angles.

    See :class:`RY` for more detail on `entanglement` which applies here too.

    The rotations of the SwapRZ ansatz can be written as

    .. math::

        R_Z(\theta) = e^{-i \theta Z}

    and

    .. math::

        R_{XX+YY}(\theta) = e^{-i \theta / 2 (X \otimes X + Y \otimes Y)}
                          \approx e^{-i \theta / 2 X \otimes X} e^{-i \theta /2 Y \otimes Y }
                          = R_{XX}(\theta) R_{YY}(\theta)

    where the approximation used comes from the Trotter expansion of the sum in the exponential.
    """

    @deprecate_arguments({'entangler_map': 'entanglement'})
    def __init__(self,
                 num_qubits: Optional[int] = None,
                 depth: int = 3,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 initial_state: Optional[InitialState] = None,
                 entangler_map: Optional[List[List[int]]] = None,  # pylint: disable=unused-argument
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            depth: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            initial_state: An `InitialState` object to prepend to the Ansatz.
                TODO deprecate this feature in favor of prepend or overloading __add__ in
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
            depth: Deprecated, use `reps` instead.
            entangler_map: Deprecated, use `entanglement` instead. This argument now also supports
                entangler maps.

        Examples:
            >>> swaprz = SwapRZ(3)  # create the variational form on 3 qubits
            >>> print(swaprz)  # show the circuit
            TODO: circuit diagram

            >>> swaprz = SwapRZ(4, entanglement='full', reps=1)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the RY variational form
            >>> qc += swaprz.to_circuit()
            >>> qc.draw()
            TODO: circuit diagram

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ry = SwapRZ(3, entangler_map, reps=2)
            >>> print(swaprz)
            TODO: circuit diagram

            >>> swaprz = SwapRZ(2, entanglement='linear', reps=1)
            >>> ry = RY(2, entanglement='full', reps=1)
            >>> my_varform = swaprz + ry
            >>> print(my_varform)
        """
        circuit = QuantumCircuit(2)
        theta = Parameter('θ')
        circuit.rxx(theta, 0, 1)
        circuit.ryy(theta, 0, 1)

        super().__init__(num_qubits=num_qubits,
                         depth=depth,
                         rotation_gates=RZGate,
                         entanglement_gates=circuit,
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

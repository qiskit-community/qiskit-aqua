# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Layers of Y+Z rotations followed by entangling gates."""

from typing import Optional, List
import numpy as np
from qiskit.circuit.library import RYRZ as RYRZCircuit
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.components.initial_states import InitialState
from .variational_form import VariationalForm


class RYRZ(VariationalForm):
    r"""The RYRZ Variational Form.

    The RYRZ trial wave function is layers of :math:`y` plus :math:`z` rotations with entanglements.
    When none of qubits are unentangled to other qubits, the number of optimizer parameters this
    form creates and uses is given by :math:`q \times (d + 1) \times 2`, where :math:`q` is the
    total number of qubits and :math:`d` is the depth of the circuit.
    Nonetheless, in some cases, if an `entangler_map` does not include all qubits, that is, some
    qubits are not entangled by other qubits. The number of parameters is reduced by
    :math:`d \times q' \times 2` where :math:`q'` is the number of unentangled qubits.
    This is because adding more parameters to the unentangled qubits only introduce overhead
    without bringing any benefit; furthermore, theoretically, applying multiple Ry and Rz gates
    in a row can be reduced to a single Ry gate and one Rz gate with the summed rotation angles.

    See :class:`RY` for more detail on `entangler_map` and `entanglement` which apply here too
    but note RYRZ only supports 'full' and 'linear' values.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 3,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 initial_state: Optional[InitialState] = None,
                 entanglement_gate: str = 'cz',
                 skip_unentangled_qubits: bool = False) -> None:
        """
        Args:
            num_qubits: Number of qubits, has a minimum value of 1.
            depth: Number of rotation layers, has a minimum value of 1.
            entangler_map: Describe the connectivity of qubits, each list pair describes
                [source, target], or None for as defined by `entanglement`.
                Note that the order is the list is the order of applying the two-qubit gate.
            entanglement: ('full' | 'linear') overridden by 'entangler_map` if its
                provided. 'full' is all-to-all entanglement, 'linear' is nearest-neighbor.
            initial_state: An initial state object
            entanglement_gate: ('cz' | 'cx')
            skip_unentangled_qubits: Skip the qubits not in the entangler_map
        """
        validate_min('num_qubits', num_qubits, 1)
        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
        validate_in_set('entanglement_gate', entanglement_gate, {'cz', 'cx'})

        if entangler_map:
            entanglement = entangler_map

        ryrz = RYRZCircuit(num_qubits,
                           entanglement_blocks=entanglement_gate,
                           reps=depth,
                           entanglement=entanglement,
                           skip_unentangled_qubits=skip_unentangled_qubits,
                           initial_state=initial_state)

        super().__init__(blueprint_circuit=ryrz)

        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        # determine the entangled qubits
        all_qubits = []
        for src, targ in self._entangler_map:
            all_qubits.extend([src, targ])
        self._entangled_qubits = sorted(list(set(all_qubits)))
        self._initial_state = initial_state
        self._entanglement_gate = entanglement_gate
        self._skip_unentangled_qubits = skip_unentangled_qubits

        self._num_parameters = ryrz.num_parameters
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._support_parameterized_circuit = True

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QEomEE algorithm """

from typing import Union, List, Optional
import logging

import numpy as np
from qiskit.aqua.operators import BaseOperator, Z2Symmetries
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomEE(ExactEigensolver):
    """ QEomEE algorithm """

    def __init__(self, operator: BaseOperator, num_orbitals: int,
                 num_particles: Union[List[int], int], qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 is_eom_matrix_symmetric: bool = True,
                 se_list: Optional[List[List[int]]] = None,
                 de_list: Optional[List[List[int]]] = None,
                 z2_symmetries: Optional[Z2Symmetries] = None,
                 untapered_op: Optional[BaseOperator] = None,
                 aux_operators: Optional[List[BaseOperator]] = None) -> None:
        """
        Args:
            operator: qubit operator
            num_orbitals:  total number of spin orbitals
            num_particles: number of particles, if it is a list,
                                        the first number is alpha and the second
                                        number if beta.
            qubit_mapping: qubit mapping type
            two_qubit_reduction: two qubit reduction is applied or not
            active_occupied: list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied: list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric: is EoM matrix symmetric
            se_list: single excitation list, overwrite the setting in active space
            de_list: double excitation list, overwrite the setting in active space
            z2_symmetries: represent the Z2 symmetries
            untapered_op: if the operator is tapered, we need untapered operator
                                         to build element of EoM matrix
            aux_operators: Auxiliary operators to be evaluated at
                                                each eigenvalue
        Raises:
            ValueError: invalid parameter
        """
        validate_min('num_orbitals', num_orbitals, 1)
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise ValueError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        super().__init__(operator, 1, aux_operators)

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles, qubit_mapping,
                                      two_qubit_reduction, active_occupied, active_unoccupied,
                                      is_eom_matrix_symmetric, se_list, de_list,
                                      z2_symmetries, untapered_op)

    def _run(self):
        super()._run()
        wave_fn = self._ret['eigvecs'][0]
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(wave_fn)
        excitation_energies = excitation_energies_gap + self._ret['energy']
        all_energies = np.concatenate(([self._ret['energy']], excitation_energies))
        self._ret['energy_gap'] = excitation_energies_gap
        self._ret['energies'] = all_energies
        self._ret['eom_matrices'] = eom_matrices
        return self._ret

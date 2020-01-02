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

""" QEomVQE algorithm """

import logging

from typing import Union, List, Optional, Callable
import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.operators import BaseOperator, Z2Symmetries
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomVQE(VQE):
    """ QEomVQE algorithm """

    def __init__(self, operator: BaseOperator, var_form: VariationalForm,
                 optimizer: Optimizer, num_orbitals: int,
                 num_particles: Union[List[int], int],
                 initial_point: Optional[np.ndarray] = None,
                 max_evals_grouped: int = 1,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 auto_conversion: bool = True, qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True, is_eom_matrix_symmetric: bool = True,
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 se_list: Optional[List[List[int]]] = None,
                 de_list: Optional[List[List[int]]] = None,
                 z2_symmetries: Optional[Z2Symmetries] = None,
                 untapered_op: Optional[BaseOperator] = None,
                 aux_operators: Optional[List[BaseOperator]] = None) -> None:
        """
        Args:
            operator: qubit operator
            var_form: parametrized variational form.
            optimizer: the classical optimization algorithm.
            num_orbitals:  total number of spin orbitals
            num_particles: number of particles, if it is a list,
                                              the first number is
                                              alpha and the second number if beta.
            initial_point: optimizer initial point, 1-D vector
            max_evals_grouped: max number of evaluations performed simultaneously
            callback: a callback that can access the intermediate data during
                                 the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion: an automatic conversion for operator and aux_operators into
                                    the type which is most suitable for the backend.

                                    - non-aer statevector_simulator: MatrixOperator
                                    - aer statevector_simulator: WeightedPauliOperator
                                    - qasm simulator or real backend:
                                        TPBGroupedWeightedPauliOperator
            qubit_mapping: qubit mapping type
            two_qubit_reduction: two qubit reduction is applied or not
            is_eom_matrix_symmetric: is EoM matrix symmetric
            active_occupied: list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied: list of unoccupied orbitals to include, indices are
                                      0 to m where m is (num_orbitals - num particles) // 2
            se_list: single excitation list, overwrite the setting in active space
            de_list: double excitation list, overwrite the setting in active space
            z2_symmetries: represent the Z2 symmetries
            untapered_op: if the operator is tapered, we need untapered operator
                                         during building element of EoM matrix
            aux_operators: Auxiliary operators to be
                                                evaluated at each eigenvalue
        """
        self._validate_qeomvqe(num_orbitals, num_particles, qubit_mapping)
        super().__init__(operator.copy(), var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles,
                                      qubit_mapping, two_qubit_reduction, active_occupied,
                                      active_unoccupied,
                                      is_eom_matrix_symmetric, se_list, de_list,
                                      z2_symmetries, untapered_op)

    def _validate_qeomvqe(self, num_orbitals: int, num_particles: Union[List[int], int],
                          qubit_mapping: str) -> None:
        if num_orbitals < 1:
            raise AquaError('Num Orbitals value {}. Minimum value allowed is 1'.format(
                num_orbitals))
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise AquaError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        if qubit_mapping not in ['jordan_wigner', 'parity', 'bravyi_kitaev']:
            raise AquaError(
                "Qubit Mapping value '{}'. Values allowed are 'jordan_wigner', 'parity', \
                'bravyi_kitaev'".format(qubit_mapping))

    def _run(self):
        super()._run()
        self._quantum_instance.circuit_summary = True
        opt_params = self._ret['opt_params']
        logger.info("opt params:\n%s", opt_params)
        wave_fn = self._var_form.construct_circuit(opt_params)
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(
            wave_fn, quantum_instance=self._quantum_instance)
        excitation_energies = excitation_energies_gap + self._ret['energy']
        all_energies = np.concatenate(([self._ret['energy']], excitation_energies))
        self._ret['energy_gap'] = excitation_energies_gap
        self._ret['energies'] = all_energies
        self._ret['eom_matrices'] = eom_matrices
        return self._ret

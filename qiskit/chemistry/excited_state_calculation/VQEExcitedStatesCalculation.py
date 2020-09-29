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

from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator)
from qiskit.chemistry.core import MolecularExcitedStatesResult
from qiskit.chemistry.excited_states_calculation import ExcitedStatesCalculation


class VQEQeomExcitedStatesCalculation(ExcitedStatesCalculation):
    """
    VQEQeomExcitesStatesCalculation
    """

    def __init__(self,
                 variational_form: var_form,
                 optimizer: optimizer,
                 backend: backend = BasicAer.get_backend('statevector_simulator'),
                 intial_state: intitial_state = None,
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:
        """

        Args:
            solver:
            transformation:
            qubit_mapping:
            two_qubit_reduction:
            freeze_core:
            orbital_reduction:
            z2symmetry_reduction:
        """

        self._variational_form = variational_form
        self._optimizer = optimizer
        self._intial_state = intial_state
        self._backend = backend
        super().__init__(transformation, qubit_mapping, two_qubit_reduction, freeze_core, orbital_reduction,
                         z2symmetry_reduction)

    def compute_excited_states(self, driver) -> MolecularExcitedStatesCalculationResult:
        """

        Compute Excited States result

        Returns:
            MolecularExcitedStatesCalculationResult

        """

        operator, aux_operators = self._transform(driver)

        #TODO This should not be like this. We need to implement a ".compute_eigenstates()" function
        #TODO Check: Make sure that the driver here exposes _num_orbitals and _num_particles

        eom_vqe = QEomVQE(operator, var_form, optimizer, num_orbitals=driver._num_orbitals,
                          num_particles=driver._num_particles, qubit_mapping=self._qubit_mapping,
                          two_qubit_reduction=self._two_qubit_reduction,
                          z2_symmetries=self._z2_symmetries, untapered_op=operator)

        quantum_instance = QuantumInstance(backend)
        raw_es_result = eom_vqe.run(quantum_instance)

        return core.process_algorithm_result(raw_es_result)

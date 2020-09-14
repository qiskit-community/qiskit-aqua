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

""" Ground state computation using Aqua minimum eigensolver """

from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator, MolecularGroundStateResult)



class MinimumEigensolverGroundStateCalculation(GroundStateCalculation):

    def __init__(self,
                 solver: Optional[MinimumEigensolver] = None,
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:

        self._solver = solver
        super().__init__(transformation, qubit_mapping, two_qubit_reduction, freeze_core, orbital_reduction,
                         z2symmetry_reduction)

    def compute_ground_state(driver) -> GroundStateCalculationResult:

        operator, aux_operators = self._transform(driver)

        aux_operators = aux_operators if self.solver.supports_aux_operators() else None

        raw_gs_result = self._solver.compute_minimum_eigenstate(operator, aux_operators)

        return core.process_algorithm_result(raw_gs_result)

    @staticmethod
    def get_default_solver(quantum_instance: Union[QuantumInstance, BaseBackend]) ->\
            Optional[Callable[[List, int, str, bool, Z2Symmetries], MinimumEigensolver]]:
        """
        Get the default solver callback that can be used with :meth:`compute_energy`
        Args:
            quantum_instance: A Backend/Quantum Instance for the solver to run on

        Returns:
            Default solver callback
        """
        def cb_default_solver(num_particles, num_orbitals,
                              qubit_mapping, two_qubit_reduction, z2_symmetries):
            """ Default solver """
            initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, z2_symmetries.sq_list)
            var_form = UCCSD(num_orbitals=num_orbitals,
                             num_particles=num_particles,
                             initial_state=initial_state,
                             qubit_mapping=qubit_mapping,
                             two_qubit_reduction=two_qubit_reduction,
                             z2_symmetries=z2_symmetries)
            vqe = VQE(var_form=var_form)
            vqe.quantum_instance = quantum_instance
            return vqe
        return cb_default_solver
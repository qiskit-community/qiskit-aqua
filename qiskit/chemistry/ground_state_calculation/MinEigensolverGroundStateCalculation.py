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

from typing import List, Optional, Callable, Union

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver, VQE
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import (TransformationType, QubitMappingType, ChemistryOperator,
                                   MolecularGroundStateResult)
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.providers import BaseBackend

class MinimumEigensolverGroundStateCalculation(GroundStateCalculation):
    """
    MinimumEigensolverGroundStateCalculation
    """

    def __init__(self, transformation, solver: Optional[MinimumEigensolver] = None) -> None:
        """
        Args:
            solver: a minimum eigensolver
            transformation:
        """

        self._solver = solver
        super().__init__(transformation)

    def compute_ground_state(self,
                             driver: BaseDriver,
                             callback: Optional[Callable[[List, int, str, bool, Z2Symmetries],
                                                         MinimumEigensolver]] = None  # WOR: callback should be in constructor, and possibly follow an interface
                             ) -> MolecularGroundStateResult:
        """Compute Ground State properties.

        Args:
            driver: A chemistry driver.
            callback: If not None will be called with the following values
                num_particles, num_orbitals, qubit_mapping, two_qubit_reduction, z2_symmetries
                in that order. This information can then be used to setup chemistry
                specific component(s) that are needed by the chosen MinimumEigensolver.
                The MinimumEigensolver can then be built and returned from this callback
                for use as the solver here.

        Returns:
            A molecular ground state result
        Raises:
            QiskitChemistryError: If no MinimumEigensolver was given and no callback is being
                                  used that could supply one instead.
        """

        if self._solver is None and callback is None:
            raise QiskitChemistryError('Minimum Eigensolver was not provided')

        operator, aux_operators = self._transformation.transform(driver)

        if callback is not None:
            #TODO We should expose all these as properties from the transformation
            num_particles = self._transformation.molecule_info['num_particles']
            num_orbitals = self._transformation.molecule_info['num_orbitals']
            z2_symmetries = self._transformation.molecule_info['z2symmetries']
            self._solver = callback(num_particles, num_orbitals,self._transformation._qubit_mapping, self._transformation._two_qubit_reduction, z2_symmetries)

        aux_operators = aux_operators if self._solver.supports_aux_operators() else None

        raw_gs_result = self._solver.compute_minimum_eigenvalue(operator, aux_operators)

        # TODO WOR: where should this post processing be coming from?
        #gsc_result = self._transformation.interpret(raw_gs_result['energy'], r['aux_values'], groundstate)  # gs = array/circuit+params
        #gsc_result.raw_result = raw_gs_results

        return raw_gs_result
        # (energy, aux_values, groundsntate)


    #class MesFactory():
    
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

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

"""The minimum eigensolver factory for ground state calculation algorithms."""

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver, VQE
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation
from qiskit.chemistry.components.initial_states import HartreeFock


class VQEUCCSDFactory:
    """A factory to construct a minimum eigensolver suitable for a qubit operator transformation.
    """

    def __init__(self, quantum_instance: QuantumInstance) -> None:
        """
        Args:
            quantum_instance: The quantum instance used in the minimum eigensolver.
        """
        self._quantum_instance = quantum_instance

    def get_solver(self, transformation: QubitOperatorTransformation) -> MinimumEigensolver:
        """Returns a minimum eigensolver, based on the qubit operator transformation.

        By default the VQE with a UCCSD wavefunction ansatz is returned.

        Args:
            transformation: The qubit operator transformation.

        Returns:
            A minimum eigensolver suitable to compute the ground state of the molecule transformed
            by ``transformation``.
        """

        num_orbitals = transformation._molecule_info['num_orbitals']
        num_particles = transformation._molecule_info['num_particles']
        qubit_mapping = transformation._qubit_mapping
        two_qubit_reduction = transformation._molecule_info['two_qubit_reduction']
        z2_symmetries = transformation._molecule_info['z2symmetries']
        initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                    two_qubit_reduction, z2_symmetries.sq_list)
        var_form = UCCSD(num_orbitals=num_orbitals,
                         num_particles=num_particles,
                         initial_state=initial_state,
                         qubit_mapping=qubit_mapping,
                         two_qubit_reduction=two_qubit_reduction,
                         z2_symmetries=z2_symmetries)
        vqe = VQE(var_form=var_form, quantum_instance=self._quantum_instance)
        return vqe

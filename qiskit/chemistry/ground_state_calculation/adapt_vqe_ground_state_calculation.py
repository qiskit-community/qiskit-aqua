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

"""
A ground state calculation employing the VQEAdapt algorithm.
"""

import logging

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.chemistry.algorithms import VQEAdapt
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.ground_state_calculation import (MinimumEigensolverGroundStateCalculation,
                                                       MESFactory)
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation

logger = logging.getLogger(__name__)


class AdaptVQEGroundStateCalculation(MinimumEigensolverGroundStateCalculation):
    """A ground state calculation employing the VQEAdapt algorithm."""

    def __init__(self,
                 transformation: QubitOperatorTransformation,
                 quantum_instance: QuantumInstance) -> None:
        """
        Args:
            transformation: TODO
            quantum_instance: TODO
        """

        super().__init__(transformation, AdaptVQEFactory(quantum_instance))


class AdaptVQEFactory(MESFactory):
    """TODO"""

    def get_solver(self, transformation: QubitOperatorTransformation) -> MinimumEigensolver:
        """TODO

        Args:
            transformation: TODO

        Returns:
            TODO
        """
        num_orbitals = transformation._molecule_info['num_orbitals']
        num_particles = transformation._molecule_info['num_particles']
        qubit_mapping = transformation._molecule_info['qubit_mapping']
        two_qubit_reduction = transformation._molecule_info['two_qubit_reduction']
        z2_symmetries = transformation._molecule_info['z2_symmetries']

        # contract variational form base
        initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping.value,
                                    two_qubit_reduction, z2_symmetries.sq_list)
        var_form_base = UCCSD(num_orbitals=num_orbitals,
                              num_particles=num_particles,
                              initial_state=initial_state,
                              qubit_mapping=qubit_mapping.value,
                              two_qubit_reduction=two_qubit_reduction,
                              z2_symmetries=z2_symmetries)

        # initialize the adaptive VQE algorithm with the specified quantum instance
        vqe = VQEAdapt(var_form_base=var_form_base, quantum_instance=self._quantum_instance)
        return vqe


# TODO we can think about moving all of the VQEAdapt code (currently in
# qiskit.chemistry.algorithms.minimum_eigen_solvers) into this class and therefore deprecate the
# qiskit.chemistry.algorithms module (if we do the same with the QEOM code)

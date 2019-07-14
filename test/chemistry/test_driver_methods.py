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

from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.aqua.algorithms.classical import ExactEigensolver
from test.chemistry.common import QiskitChemistryTestCase


class TestDriverMethods(QiskitChemistryTestCase):
    """Common driver tests. For H2 @ 0.735, sto3g"""

    def setup(self):
        super().setUp()

    LIH = 'LI 0 0 0; H 0 0 1.6'
    OH = 'O 0 0 0; H 0 0 0.9697'

    ref_energies = {
        'lih': -7.882,
        'oh': -74.387
    }

    ref_dipoles = {
        'lih': 1.818,
        'oh': 0.4615
    }

    @staticmethod
    def _run_driver(driver, transformation=TransformationType.FULL,
                    qubit_mapping=QubitMappingType.JORDAN_WIGNER, two_qubit_reduction=False,
                    freeze_core=True):
        qmolecule = driver.run()

        core = Hamiltonian(transformation=transformation,
                           qubit_mapping=qubit_mapping,
                           two_qubit_reduction=two_qubit_reduction,
                           freeze_core=freeze_core,
                           orbital_reduction=[])

        qubit_op, aux_ops = core.run(qmolecule)

        ee = ExactEigensolver(qubit_op, aux_operators=aux_ops, k=1)
        lines, result = core.process_algorithm_result(ee.run())
        # print(*lines, sep='\n')

        return result

    def _assert_energy(self, result, mol):
        self.assertAlmostEqual(self.ref_energies[mol], result['energy'], places=3)

    def _assert_energy_and_dipole(self, result, mol):
        self._assert_energy(result, mol)
        self.assertAlmostEqual(self.ref_dipoles[mol], result['total_dipole_moment'], places=3)

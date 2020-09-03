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

""" Test Harmonic Integrals """

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np

from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.chemistry import BosonicOperator
from qiskit.chemistry.drivers import GaussianLogResult


class TestHarmonicIntegrals(QiskitChemistryTestCase):
    """Hamiltonian in harmonic basis tests."""

    def setUp(self):
        super().setUp()

        self.reference_energy = 2539.259482550559
        self.gaussian_log_data = GaussianLogResult(
            self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log'))

    def test_compute_modes(self):
        """ test for computing the general hamiltonian from the gaussian log data"""
        reference = [[605.3643675, 1, 1],
                     [-605.3643675, -1, -1],
                     [340.5950575, 2, 2],
                     [-340.5950575, -2, -2],
                     [163.7595125, 3, 3],
                     [-163.7595125, -3, -3],
                     [163.7595125, 4, 4],
                     [-163.7595125, -4, -4],
                     [-89.09086530649508, 2, 1, 1],
                     [-15.590557244410897, 2, 2, 2],
                     [44.01468537435673, 3, 2, 1],
                     [21.644966371722838, 3, 3, 2],
                     [-78.71701132125833, 4, 2, 1],
                     [17.15529085952822, 4, 3, 2],
                     [6.412754934114705, 4, 4, 2],
                     [1.6512647916666667, 1, 1, 1, 1],
                     [5.03965375, 2, 2, 1, 1],
                     [0.43840625000000005, 2, 2, 2, 2],
                     [-2.4473854166666666, 3, 1, 1, 1],
                     [-3.73513125, 3, 2, 2, 1],
                     [-6.3850425, 3, 3, 1, 1],
                     [-2.565723125, 3, 3, 2, 2],
                     [1.7778641666666666, 3, 3, 3, 1],
                     [0.6310235416666666, 3, 3, 3, 3],
                     [4.376968333333333, 4, 1, 1, 1],
                     [6.68000625, 4, 2, 2, 1],
                     [-5.82197125, 4, 3, 1, 1],
                     [-2.86914875, 4, 3, 2, 2],
                     [-9.53873625, 4, 3, 3, 1],
                     [1.3534904166666666, 4, 3, 3, 3],
                     [-4.595841875, 4, 4, 1, 1],
                     [-1.683979375, 4, 4, 2, 2],
                     [5.3335925, 4, 4, 3, 1],
                     [-0.551021875, 4, 4, 3, 3],
                     [-3.1795791666666666, 4, 4, 4, 1],
                     [1.29536, 4, 4, 4, 3],
                     [0.20048104166666667, 4, 4, 4, 4]]

        result = self.gaussian_log_data._compute_modes()

        check_indices = np.random.randint(0, high=len(reference), size=10)
        for idx in check_indices:
            for i in range(len(reference[idx])):
                self.assertAlmostEqual(reference[idx][i], result[idx][i], places=6)

    def test_harmonic_basis(self):
        """test for obtaining the hamiltonian in the harmonic basis"""

        num_modals = 2
        hamiltonian_in_harmonic_basis = \
            self.gaussian_log_data.compute_harmonic_modes(num_modals, truncation_order=2)
        basis = [num_modals, num_modals, num_modals, num_modals]  # 4 modes and 2 modals per mode
        bos_op = BosonicOperator(hamiltonian_in_harmonic_basis, basis)
        qubit_op = bos_op.mapping('direct', threshold=1e-5)
        algo = NumPyEigensolver(qubit_op, k=100)
        result = algo.run()
        vecs = result['eigenstates']
        energies = result['eigenvalues']
        gs_energy = bos_op.ground_state_energy(vecs, energies)
        self.assertAlmostEqual(gs_energy, self.reference_energy, places=6)


if __name__ == '__main__':
    unittest.main()

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

import unittest
from functools import partial

import chemistry.code.test.test_data as td
import numpy as np
from chemistry.code.vibronic_structure_fd import VibronicStructure1DFD
from qiskit.chemistry.algorithms.pes_samplers.potentials.harmonic_potential import HarmonicPotential
from qiskit.chemistry.algorithms.pes_samplers.potentials.morse_potential import MorsePotential
from qiskit.chemistry.drivers import Molecule


# TODO Fix this test
class TestFDVibronic(unittest.TestCase):
    def create_test_molecule(self):
        stretch = partial(Molecule.absolute_stretching,
                          kwargs={'atom_pair': (1, 0)})
        m = Molecule(geometry=[['H', [0., 0., 0.]], ['D', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 3.444946E-27],
                     spins=[1 / 2, 1])
        return m

    def test_with_morse(self):
        m = self.create_test_molecule()

        M = MorsePotential(m)

        xdata = np.array(td.xdata_angstrom)
        ydata = np.array(td.ydata_hartree)

        M.fit_to_data(xdata, ydata)

        VS = VibronicStructure1DFD(m, M)

        N = np.array(range(2, 8))
        vib_levels = VS.vibrational_energy_level(N)
        vib_levels_ref = M.vibrational_energy_level(N)
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref, decimal=5)

    def test_with_harmonic(self):
        m = self.create_test_molecule()

        H = HarmonicPotential(m)

        xdata = np.array(td.xdata_angstrom)
        ydata = np.array(td.ydata_hartree)

        H.fit_to_data(xdata, ydata)

        VS = VibronicStructure1DFD(m, H)

        N = np.array(range(2, 8))
        vib_levels = VS.vibrational_energy_level(N)
        vib_levels_ref = H.vibrational_energy_level(N)
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref, decimal=4)


if __name__ == '__main__':
    unittest.main()

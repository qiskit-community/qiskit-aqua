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

""" Test Potential """

import unittest
from functools import partial
import numpy as np

from qiskit.chemistry.algorithms.pes_samplers.potentials.harmonic_potential import HarmonicPotential
from qiskit.chemistry.algorithms.pes_samplers.potentials.morse_potential import MorsePotential
from qiskit.chemistry.constants import HARTREE_TO_J_PER_MOL
from qiskit.chemistry.drivers.molecule import Molecule


class TestPotential(unittest.TestCase):
    """ Test Potential """

    @staticmethod
    def create_test_molecule():
        """ create test molecule """
        stretch = partial(Molecule.absolute_stretching,
                          kwargs={'atom_pair': (1, 0)})
        m = Molecule(geometry=[['H', [0., 0., 0.]], ['D', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 3.444946E-27])
        return m

    def test_morse(self):
        """ test morse """
        xdata = np.array([0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15,
                          3.45, 3.75, 4.05, 4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05,
                          1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05,
                          4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95,
                          2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65, 4.95,
                          5.25])

        ydata = np.array([-2254757.5348101, -2746067.46608231, -2664406.49829366,
                          -2611323.75276296, -2502198.92978322, -2417457.48952287,
                          -2390778.71123391, -2379482.70907613, -2373850.72354504,
                          -2361426.93801724, -2369992.6305902, -2363833.07716161,
                          -2360577.93019891, -2356002.65576262, -2355574.41051646,
                          -2357254.94032554, -2351656.71871981, -2308055.75509618,
                          -2797576.98597419, -2715367.76135088, -2616523.58105343,
                          -2498053.2658529, -2424288.88205414, -2393385.83237565,
                          -2371800.12956182, -2353202.82294735, -2346873.32092711,
                          -2343485.8487826, -2342937.74947792, -2350276.02096954,
                          -2347674.75469199, -2346912.78218669, -2339886.28877723,
                          -2353456.10489755, -2359599.85281831, -2811321.68662548,
                          -2763866.98837641, -2613385.92519959, -2506804.00364042,
                          -2419329.49702063, -2393428.68052976, -2374166.67617163,
                          -2352961.35574553, -2344972.64297329, -2356294.5588125,
                          -2341396.63369969, -2337344.83138146, -2339793.71365995,
                          -2335667.95101689, -2327347.45385524, -2341367.28061372])

        ydata_hartree = ydata / HARTREE_TO_J_PER_MOL

        xdata_angstrom = xdata

        m = self.create_test_molecule()

        morse = MorsePotential(m)

        xdata = np.array(xdata_angstrom)
        ydata = np.array(ydata_hartree)

        morse.fit(xdata, ydata)

        minimal_energy_distance = morse.get_equilibrium_geometry()
        minimal_energy = morse.eval(minimal_energy_distance)
        wave_number = morse.wave_number()

        result = np.array([minimal_energy_distance, minimal_energy, wave_number])
        benchmark = np.array([0.8106703001726382, -1.062422610690636, 3800.7855102410026])
        np.testing.assert_array_almost_equal(result, benchmark, decimal=4)

        radia = np.array([0.5, 1, 1.5, 2])
        hartrees = np.array([-0.94045495, -1.04591482, -0.96876003, -0.92400906])
        np.testing.assert_array_almost_equal(hartrees, morse.eval(radia), decimal=4)

        vib_levels = []
        for level in range(2, 8):
            vib_levels.append(morse.vibrational_energy_level(level))
        vib_levels = np.array(vib_levels)
        vib_levels_ref = np.array([0.04052116451981064, 0.05517676610999135,
                                   0.06894501671860434, 0.08182591634564956,
                                   0.09381946499112709, 0.10492566265503685])
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref, decimal=4)

    def test_harmonic(self):
        """ test harmonic """
        xdata = np.array([0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15,
                          3.45, 3.75, 4.05, 4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05,
                          1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05,
                          4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95,
                          2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65, 4.95,
                          5.25])

        ydata = np.array([-2254757.5348101, -2746067.46608231, -2664406.49829366,
                          -2611323.75276296, -2502198.92978322, -2417457.48952287,
                          -2390778.71123391, -2379482.70907613, -2373850.72354504,
                          -2361426.93801724, -2369992.6305902, -2363833.07716161,
                          -2360577.93019891, -2356002.65576262, -2355574.41051646,
                          -2357254.94032554, -2351656.71871981, -2308055.75509618,
                          -2797576.98597419, -2715367.76135088, -2616523.58105343,
                          -2498053.2658529, -2424288.88205414, -2393385.83237565,
                          -2371800.12956182, -2353202.82294735, -2346873.32092711,
                          -2343485.8487826, -2342937.74947792, -2350276.02096954,
                          -2347674.75469199, -2346912.78218669, -2339886.28877723,
                          -2353456.10489755, -2359599.85281831, -2811321.68662548,
                          -2763866.98837641, -2613385.92519959, -2506804.00364042,
                          -2419329.49702063, -2393428.68052976, -2374166.67617163,
                          -2352961.35574553, -2344972.64297329, -2356294.5588125,
                          -2341396.63369969, -2337344.83138146, -2339793.71365995,
                          -2335667.95101689, -2327347.45385524, -2341367.28061372])

        ydata_hartree = ydata / HARTREE_TO_J_PER_MOL

        xdata_angstrom = xdata

        m = self.create_test_molecule()

        harmonic = HarmonicPotential(m)

        xdata = np.array(xdata_angstrom)
        ydata = np.array(ydata_hartree)

        xdata, ydata = HarmonicPotential.process_fit_data(xdata, ydata)
        harmonic.fit(xdata, ydata)

        minimal_energy_distance = harmonic.get_equilibrium_geometry()
        minimal_energy = harmonic.eval(minimal_energy_distance)
        wave_number = harmonic.wave_number()

        result = np.array([minimal_energy_distance, minimal_energy, wave_number])
        benchmark = np.array([0.8792058944654566, -1.0678714520398802, 4670.969897517367])
        np.testing.assert_array_almost_equal(result, benchmark)

        radia = np.array([0.5, 1, 1.5, 2])
        hartrees = np.array([-0.92407434, -1.05328024, -0.68248613, 0.18830797])
        np.testing.assert_array_almost_equal(hartrees, harmonic.eval(radia))

        vib_levels = []
        for level in range(2, 8):
            vib_levels.append(harmonic.vibrational_energy_level(level))
        vib_levels = np.array(vib_levels)
        vib_levels_ref = np.array([0.053206266711245426, 0.07448877339574358,
                                   0.09577128008024177, 0.11705378676473993,
                                   0.13833629344923812, 0.15961880013373628])
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref)


if __name__ == '__main__':
    unittest.main()

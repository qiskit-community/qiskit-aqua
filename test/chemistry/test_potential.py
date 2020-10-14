import unittest

import numpy as np

import qiskit.chemistry.constants as const

from functools import partial
import matplotlib.pyplot as plt


from qiskit.chemistry.algorithms.pes_samplers.potentials.morse_potential import MorsePotential
from qiskit.chemistry.algorithms.pes_samplers.potentials.harmonic_potential import HarmonicPotential
from qiskit.chemistry.drivers.molecule import Molecule

from qiskit.chemistry.constants import HARTREE_TO_J_PER_MOL

class TestPotential(unittest.TestCase):

    def create_test_molecule(self):
        stretch = partial(Molecule.absolute_stretching,
                          kwargs={'atom_pair': (1, 0)})
        m = Molecule(geometry=[['H', [0., 0., 0.]], ['D', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 3.444946E-27])
        return m

    def test_morse(self):

        self._xdata = np.array([0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15,
                   3.45, 3.75, 4.05, 4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05,
                   1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05,
                   4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95,
                   2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65, 4.95,
                   5.25])

        self._ydata = np.array([-2254757.5348101, -2746067.46608231, -2664406.49829366,
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

        self.ydata_hartree = self._ydata / HARTREE_TO_J_PER_MOL
        self.ydata_j_per_mol = self._ydata

        self.xdata_angstrom = self._xdata
        
        m = self.create_test_molecule()

        M = MorsePotential(m)

        xdata = np.array(self.xdata_angstrom)
        ydata = np.array(self.ydata_hartree)

        M.fit_to_data(xdata, ydata)

        #self.plot_potential(xdata, ydata, M)

        minimalEnergyDistance = M.get_equilibrium_geometry()
        minimalEnergy = M.eval(minimalEnergyDistance)
        waveNumber = M.wave_number()

        result = np.array([minimalEnergyDistance, minimalEnergy, waveNumber])
        benchmark = np.array([0.8106703001726382,
                              -1.062422610690636, 3800.7855102410026])
        np.testing.assert_array_almost_equal(result, benchmark)

        radia = np.array([0.5, 1, 1.5, 2])
        hartrees = np.array(
            [-0.94045495, -1.04591482, -0.96876003, -0.92400906])
        np.testing.assert_array_almost_equal(hartrees, M.eval(radia))

        vib_levels = []
        for N in range(2, 8):
            vib_levels.append(M.vibrational_energy_level(N))
        vib_levels = np.array(vib_levels)
        vib_levels_ref = np.array([0.04052116451981064, 0.05517676610999135,
                                   0.06894501671860434, 0.08182591634564956,
                                   0.09381946499112709, 0.10492566265503685])
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref)

    def test_harmonic(self):

        self._xdata = np.array([0.45, 0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15,
                                3.45, 3.75, 4.05, 4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05,
                                1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05,
                                4.35, 4.65, 4.95, 5.25, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95,
                                2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65, 4.95,
                                5.25])

        self._ydata = np.array([-2254757.5348101, -2746067.46608231, -2664406.49829366,
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

        self.ydata_hartree = self._ydata / HARTREE_TO_J_PER_MOL
        self.ydata_j_per_mol = self._ydata

        self.xdata_angstrom = self._xdata

        m = self.create_test_molecule()

        H = HarmonicPotential(m)

        xdata = np.array(self.xdata_angstrom)
        ydata = np.array(self.ydata_hartree)

        H.fit_to_data(xdata, ydata)

        #self.plot_potential(xdata, ydata, H)

        minimalEnergyDistance = H.get_equilibrium_geometry()
        minimalEnergy = H.eval(minimalEnergyDistance)
        waveNumber = H.wave_number()

        result = np.array([minimalEnergyDistance, minimalEnergy, waveNumber])
        benchmark = np.array([0.8792058944654566,
                              -1.0678714520398802, 4670.969897517367])
        np.testing.assert_array_almost_equal(result, benchmark)

        radia = np.array([0.5, 1, 1.5, 2])
        hartrees = np.array(
            [-0.92407434, -1.05328024, -0.68248613, 0.18830797])
        np.testing.assert_array_almost_equal(hartrees,
                                             H.eval(radia))

        vib_levels = []
        for N in range(2, 8):
            vib_levels.append(H.vibrational_energy_level(N))
        vib_levels = np.array(vib_levels)
        vib_levels_ref = np.array([0.053206266711245426, 0.07448877339574358,
                                   0.09577128008024177, 0.11705378676473993,
                                   0.13833629344923812, 0.15961880013373628])
        np.testing.assert_array_almost_equal(vib_levels, vib_levels_ref)


if __name__ == '__main__':
    unittest.main()

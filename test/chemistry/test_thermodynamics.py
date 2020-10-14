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
import chemistry.code.thermodynamics as thermo
import numpy as np
from chemistry.code.molecule import Molecule
from chemistry.code.morse_potential import MorsePotential
from chemistry.code.partition_function import DiatomicPartitionFunction


# TODO Fix this test

class TestThermodynamics(unittest.TestCase):
    def create_test_molecule(self):
        stretch = partial(Molecule.absolute_stretching,
                          kwargs={'atom_pair': (1, 0)})
        m = Molecule(geometry=[['H', [0., 0., 0.]], ['D', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 3.444946E-27],
                     spins=[1 / 2, 1])
        return m

    def perform_test(self, thermo_func_name, benchmark):
        pf_callable = self.pf_callable
        td_class = self.td_class

        temps = np.array([10, 50, 100, 200])

        from_df_callable = getattr(thermo, thermo_func_name)(pf_callable, temps)
        from_class = getattr(td_class, thermo_func_name)(temps)

        np.testing.assert_array_equal(from_class, from_df_callable)
        np.testing.assert_array_almost_equal(from_class, benchmark)

    def test_thermo(self):
        m = self.create_test_molecule()

        morse = MorsePotential(m)

        xdata = np.array(td.xdata_angstrom)
        ydata = np.array(td.ydata_hartree)

        morse.fit_to_data(xdata, ydata)

        partition_function = DiatomicPartitionFunction(m, morse, morse)

        pressure = 102523

        self.pf_callable = partition_function.get_default_callable(pressure=pressure)
        self.td_class = thermo.Thermodynamics(partition_function, pressure)

        benchmark = np.array([22129.20266631, 19079.17137585, 13863.05240428, 1405.13250492])
        self.perform_test('helmholtz_free_energy', benchmark)

        benchmark = np.array([22567.29835223, 23309.38074915, 24363.27230243, 26445.90231315])
        self.perform_test('thermodynamic_energy', benchmark)

        benchmark = np.array([43.80956859, 84.60418747, 105.00219898, 125.20384904])
        self.perform_test('entropy', benchmark)

        benchmark = np.array([22650.44297369, 23725.10385642, 25194.71851698, 28108.79474224])
        self.perform_test('enthalpy', benchmark)

        benchmark = np.array([22212.34728776, 19494.89448313, 14694.49861883, 3068.02493402])
        self.perform_test('gibbs_free_energy', benchmark)

        benchmark = np.array([12.52720256, 21.50674114, 20.88104447, 20.80339556])
        self.perform_test('constant_volume_heat_capacity', benchmark)

        benchmark = np.array([20.84166472, 29.8212033, 29.19550663, 29.11785772])
        self.perform_test('constant_pressure_heat_capacity', benchmark)

    def test_non_differentiable_callable(self):
        m = self.create_test_molecule()

        morse = MorsePotential(m)

        xdata = np.array(td.xdata_angstrom)
        ydata = np.array(td.ydata_hartree)

        morse.fit_to_data(xdata, ydata)

        partition_function = DiatomicPartitionFunction(m, morse, morse)

        pressure = 102523
        temps = np.array([10, 50, 100, 200])
        vib = partition_function.get_partition(part='vib', pressure=pressure)
        benchmark = np.array([22442.53040649, 22442.53040649, 22442.53040649, 22442.53040672])
        diff = thermo.thermodynamic_energy(vib, temps)
        n_diff = thermo.thermodynamic_energy(lambda t_value: vib(t_value), temps)

        np.testing.assert_array_almost_equal(diff, benchmark)
        np.testing.assert_array_almost_equal(diff, n_diff, decimal=5)


if __name__ == '__main__':
    unittest.main()

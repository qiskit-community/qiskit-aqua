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

""" Test Partition Function """

import unittest
from functools import partial
import numpy as np

import chemistry.code.test.test_data as td

from chemistry.code.partition_function import DifferentiableFunction
from chemistry.code.partition_function import DiatomicPartitionFunction
from chemistry.code.molecule import Molecule
from chemistry.code.morse_potential import MorsePotential

# TODO Fix this test


class TestPartitionFunction(unittest.TestCase):
    """ Test Partition Function """
    def create_test_molecule(self):
        """ create test molecule """
        stretch = partial(Molecule.absolute_stretching,
                          kwargs={'atom_pair': (1, 0)})
        m = Molecule(geometry=[['H', [0., 0., 0.]], ['D', [0., 0., 1.]]],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 3.444946E-27],
                     spins=[1 / 2, 1])
        return m

    def test_partition_function(self):
        """ test partition function """
        m = self.create_test_molecule()

        M = MorsePotential(m)

        xdata = np.array(td.xdata_angstrom)
        ydata = np.array(td.ydata_hartree)

        M.fit_to_data(xdata, ydata)

        P = DiatomicPartitionFunction(m, M, M)

        pressure = 102523
        temps = np.array([10, 50, 100, 200])

        trans = P.get_partition(part="trans", pressure=pressure)
        with self.assertWarns(RuntimeWarning):
            P.get_partition(part="trans", split='para', pressure=pressure)
        log_trans = np.array(
            [3.76841035, 7.79200513, 9.52487308, 11.25774103])
        np.testing.assert_array_almost_equal(log_trans, np.log(trans(temps)))

        vib = P.get_partition(part="vib", pressure=pressure)
        log_vib = np.array(
            [-269.92161476, -53.98432295, -26.99216148, -13.49608074])
        np.testing.assert_array_almost_equal(log_vib, np.log(vib(temps)))

        rot = P.get_partition(part="rot", pressure=pressure)
        log_rot = np.array(
            [5.63843874e-05, 2.98377556e-01, 7.93867745e-01, 1.39334663e+00])
        np.testing.assert_array_almost_equal(log_rot, np.log(rot(temps)))

    def test_differentiable_function(self):
        def f(x, c):
            return x**3 + np.exp(2*x) + np.log(x) + np.sin(x) + c

        def df(x, c):
            return 3*x**2 + 2*np.exp(2*x) + 1/x + np.cos(x)

        F = DifferentiableFunction(f, argument_name='x')
        F_analytic = DifferentiableFunction(f, derivative=df)
        F_dc = DifferentiableFunction(f, argument_name='c')

        self.assertEqual(F(1, 2), f(1, 2))
        self.assertAlmostEqual(F.D(1, 2), df(1, 2), places=6)
        self.assertEqual(F_analytic.D(1, 2), df(1, 2))
        self.assertAlmostEqual(F_dc.D(1, 2), 1.0)


if __name__ == '__main__':
    unittest.main()

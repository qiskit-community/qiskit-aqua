import unittest
from scipy import interpolate
from qiskit.aqua.components.extrapolation_pass_managers import RedundantCNOT
from qiskit.aqua.components.extrapolation_pass_managers import RichardsonExtrapolator
from qiskit.transpiler import PassManager


class RichardsonExtrapolatorTest(unittest.TestCase):

    def setUp(self):
        self.extrapolator = lambda x, y, val: interpolate.interp1d(x, y, fill_value='extrapolate')(val)
        self.order_parameters = [0, 1, 2, 3]
        self.extrapolated_value = -1
        self.pass_mans = [PassManager(RedundantCNOT(i)) for i in self.order_parameters]

        self.richardson_extrapolator = RichardsonExtrapolator(
            self.extrapolator,
            self.pass_mans,
            self.order_parameters,
            self.extrapolated_value
        )

    def test_extrapolation(self):
        energies = [3*x+2 for x in self.order_parameters]

        result = self.richardson_extrapolator.extrapolate(energies)

        self.assertAlmostEqual(
            -1.0,
            result,
            places=5,
            msg='Found interpolated value {}, expected {}'.format(result, -1.0)
        )


if __name__ == '__main__':
    unittest.main()

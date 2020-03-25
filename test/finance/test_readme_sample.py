# -*- coding: utf-8 -*-

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

"""
Code inside the test is the finance sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

from test.finance import QiskitFinanceTestCase


class TestReadmeSample(QiskitFinanceTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """ overloads print to log values """
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        import numpy as np
        from qiskit import BasicAer
        from qiskit.aqua.algorithms import AmplitudeEstimation
        from qiskit.aqua.components.uncertainty_models import MultivariateNormalDistribution
        from qiskit.finance.components.uncertainty_problems import FixedIncomeExpectedValue

        # Create a suitable multivariate distribution
        mvnd = MultivariateNormalDistribution(num_qubits=[2, 2],
                                              low=[0, 0], high=[0.12, 0.24],
                                              mu=[0.12, 0.24], sigma=0.01 * np.eye(2))

        # Create fixed income component
        fixed_income = FixedIncomeExpectedValue(mvnd, np.eye(2), np.zeros(2),
                                                cash_flow=[1.0, 2.0], c_approx=0.125)

        # Set number of evaluation qubits (samples)
        num_eval_qubits = 5

        # Construct and run amplitude estimation
        algo = AmplitudeEstimation(num_eval_qubits, fixed_income)
        result = algo.run(BasicAer.get_backend('statevector_simulator'))

        print('Estimated value:\t%.4f' % result['estimation'])
        print('Probability:    \t%.4f' % result['max_probability'])

        # ----------------------------------------------------------------------

        self.assertAlmostEqual(result['estimation'], 2.46, places=4)
        self.assertAlmostEqual(result['max_probability'], 0.8487, places=4)


if __name__ == '__main__':
    unittest.main()

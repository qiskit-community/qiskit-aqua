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
Code inside the test is the aqua sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest
from test.aqua import QiskitAquaTestCase


class TestReadmeSample(QiskitAquaTestCase):
    """Test sample code from readme"""

    def setUp(self):
        super().setUp()
        try:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=unused-import
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """ overloads print to log values """
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit import Aer
        from qiskit.aqua.components.oracles import LogicalExpressionOracle
        from qiskit.aqua.algorithms import Grover

        sat_cnf = """
        c Example DIMACS 3-sat
        p cnf 3 5
        -1 -2 -3 0
        1 -2 3 0
        1 2 -3 0
        1 -2 -3 0
        -1 2 3 0
        """

        backend = Aer.get_backend('qasm_simulator')
        oracle = LogicalExpressionOracle(sat_cnf)
        algorithm = Grover(oracle)
        result = algorithm.run(backend)
        print(result.assignment)

        # ----------------------------------------------------------------------

        valid_set = [[-1, -2, -3], [1, -2, 3], [1, 2, -3]]
        found = result.assignment in valid_set
        self.assertTrue(found, "Result {} is not in valid set {}".
                        format(result.assignment, valid_set))


if __name__ == '__main__':
    unittest.main()

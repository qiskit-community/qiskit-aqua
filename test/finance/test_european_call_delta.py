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

""" Test European Call Expected Value uncertainty problem """

import unittest
from test.finance import QiskitFinanceTestCase

from qiskit.circuit.library import IntegerComparator
from qiskit.finance.applications import EuropeanCallDelta
from qiskit.quantum_info import Operator


class TestEuropeanCallDelta(QiskitFinanceTestCase):
    """Tests European Call Expected Value uncertainty problem """

    def test_circuit(self):
        """Test the expected circuit.

        If it equals the correct ``IntegerComparator`` we know the circuit is correct.
        """
        num_qubits = 3
        strike_price = 0.5
        bounds = (0, 2)
        ecd = EuropeanCallDelta(num_qubits, strike_price, bounds)

        # map strike_price to a basis state
        x = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (2 ** num_qubits - 1)
        comparator = IntegerComparator(num_qubits, x)

        self.assertTrue(Operator(ecd).equiv(comparator))


if __name__ == '__main__':
    unittest.main()

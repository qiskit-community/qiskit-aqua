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

"""Test the quantum natural gradient."""

# pylint: disable=invalid-name

import unittest
from test.aqua import QiskitAquaTestCase

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.aqua.operators.gradients import NaturalGradient


class TestNaturalGradient(QiskitAquaTestCase):
    """Test the quantum natural gradient."""

    def test_gradient_circuits(self):
        """Test the gradient circuits."""

        qc = QuantumCircuit(3)
        a, b = Parameter('A'), Parameter('B')
        qc.x(1)
        qc.h(2)
        qc.cx(0, 1)
        qc.ry(a, 0)
        qc.rz(b, 2)
        qc.z(2)
        qc.h(1)
        qc.rx(a, 0)

        qng = NaturalGradient(qc)
        for param in [a, b]:
            qng.compute_gradient(param)


if __name__ == '__main__':
    unittest.main()

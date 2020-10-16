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

""" Test Bosonic Operator """

import unittest
from test.chemistry import QiskitChemistryTestCase

import numpy as np

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.chemistry import BosonicOperator


class TestBosonicOperator(QiskitChemistryTestCase):
    """Bosonic Operator tests."""

    CO2_B3LYP_ccpVDZ_4MODES_2MODALS = [[[[[0, 0, 0]], 1215.682529375],
                                        [[[0, 1, 1]], 3656.9551768750007],
                                        [[[1, 0, 0]], 682.5053337500001],
                                        [[[1, 0, 1]], -46.77167173323271],
                                        [[[1, 1, 0]], -46.77167173323271],
                                        [[[1, 1, 1]], 2050.1464387500005],
                                        [[[2, 0, 0]], 329.41209562500006],
                                        [[[2, 1, 1]], 992.0224281250003],
                                        [[[3, 0, 0]], 328.12046812500006],
                                        [[[3, 1, 1]], 985.5642906250002]],
                                       [[[[1, 0, 0], [0, 0, 0]], 5.039653750000002],
                                        [[[1, 0, 0], [0, 1, 1]], 15.118961250000009],
                                        [[[1, 0, 1], [0, 0, 0]], -89.0908653064951],
                                        [[[1, 0, 1], [0, 1, 1]], -267.27259591948535],
                                        [[[1, 1, 0], [0, 0, 0]], -89.0908653064951],
                                        [[[1, 1, 0], [0, 1, 1]], -267.27259591948535],
                                        [[[1, 1, 1], [0, 0, 0]], 15.118961250000009],
                                        [[[1, 1, 1], [0, 1, 1]], 45.35688375000003],
                                        [[[2, 0, 0], [0, 0, 0]], -6.3850425000000035],
                                        [[[2, 0, 0], [0, 1, 1]], -19.15512750000001],
                                        [[[2, 0, 0], [1, 0, 0]], -2.5657231250000008],
                                        [[[2, 0, 0], [1, 0, 1]], 21.644966371722845],
                                        [[[2, 0, 0], [1, 1, 0]], 21.644966371722845],
                                        [[[2, 0, 0], [1, 1, 1]], -7.697169375000003],
                                        [[[2, 0, 1], [0, 0, 1]], -2.0085637500000004],
                                        [[[2, 0, 1], [0, 1, 0]], -2.0085637500000004],
                                        [[[2, 1, 0], [0, 0, 1]], -2.0085637500000004],
                                        [[[2, 1, 0], [0, 1, 0]], -2.0085637500000004],
                                        [[[2, 1, 1], [0, 0, 0]], -19.15512750000001],
                                        [[[2, 1, 1], [0, 1, 1]], -57.46538250000003],
                                        [[[2, 1, 1], [1, 0, 0]], -7.697169375000004],
                                        [[[2, 1, 1], [1, 0, 1]], 64.93489911516855],
                                        [[[2, 1, 1], [1, 1, 0]], 64.93489911516855],
                                        [[[2, 1, 1], [1, 1, 1]], -23.091508125000015],
                                        [[[3, 0, 0], [0, 0, 0]], -4.595841875000001],
                                        [[[3, 0, 0], [0, 1, 1]], -13.787525625000006],
                                        [[[3, 0, 0], [1, 0, 0]], -1.683979375000001],
                                        [[[3, 0, 0], [1, 0, 1]], 6.412754934114709],
                                        [[[3, 0, 0], [1, 1, 0]], 6.412754934114709],
                                        [[[3, 0, 0], [1, 1, 1]], -5.051938125000003],
                                        [[[3, 0, 0], [2, 0, 0]], -0.5510218750000002],
                                        [[[3, 0, 0], [2, 1, 1]], -1.6530656250000009],
                                        [[[3, 0, 1], [0, 0, 1]], 3.5921675000000004],
                                        [[[3, 0, 1], [0, 1, 0]], 3.5921675000000004],
                                        [[[3, 0, 1], [2, 0, 1]], 7.946551250000004],
                                        [[[3, 0, 1], [2, 1, 0]], 7.946551250000004],
                                        [[[3, 1, 0], [0, 0, 1]], 3.5921675000000004],
                                        [[[3, 1, 0], [0, 1, 0]], 3.5921675000000004],
                                        [[[3, 1, 0], [2, 0, 1]], 7.946551250000004],
                                        [[[3, 1, 0], [2, 1, 0]], 7.946551250000004],
                                        [[[3, 1, 1], [0, 0, 0]], -13.787525625000006],
                                        [[[3, 1, 1], [0, 1, 1]], -41.362576875000016],
                                        [[[3, 1, 1], [1, 0, 0]], -5.051938125000002],
                                        [[[3, 1, 1], [1, 0, 1]], 19.238264802344126],
                                        [[[3, 1, 1], [1, 1, 0]], 19.238264802344126],
                                        [[[3, 1, 1], [1, 1, 1]], -15.15581437500001],
                                        [[[3, 1, 1], [2, 0, 0]], -1.6530656250000009],
                                        [[[3, 1, 1], [2, 1, 1]], -4.959196875000003]]]

    def setUp(self):
        super().setUp()

        self.reference_energy = 2536.4879763624226

        self.basis = [2, 2, 2, 2]  # 4 modes and 2 modals per mode
        self.bos_op = BosonicOperator(self.CO2_B3LYP_ccpVDZ_4MODES_2MODALS, self.basis)

    def test_mapping(self):
        """ mapping test """
        qubit_op = self.bos_op.mapping('direct', threshold=1e-5)
        algo = NumPyMinimumEigensolver(
            qubit_op, filter_criterion=self.bos_op.direct_mapping_filtering_criterion)
        result = algo.run()
        gs_energy = np.real(result['eigenvalue'])

        self.assertAlmostEqual(gs_energy, self.reference_energy, places=4)


if __name__ == '__main__':
    unittest.main()

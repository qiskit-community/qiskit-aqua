# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
import unittest
import numpy as np
from qiskit.aqua.utils import random_unitary

from test.common import QiskitChemistryTestCase
from qiskit.chemistry import FermionicOperator, QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType


def h2_transform_slow(h2, unitary_matrix):
    """
    Transform h2 based on unitry matrix, and overwrite original property.
    #MARK: A naive implementation based on MATLAB implementation.
    Args:
        unitary_matrix (numpy 2-D array, np.float or np.complex):
                    Unitary matrix for h2 transformation.
    """
    num_modes = unitary_matrix.shape[0]
    temp1 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp2 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp3 = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    temp_ret = np.zeros((num_modes, num_modes, num_modes, num_modes), dtype=unitary_matrix.dtype)
    unitary_matrix_dagger = np.conjugate(unitary_matrix)
    for a in range(num_modes):
        for i in range(num_modes):
            temp1[a, :, :, :] += unitary_matrix_dagger[i, a] * h2[i, :, :, :]
        for b in range(num_modes):
            for j in range(num_modes):
                temp2[a, b, :, :] += unitary_matrix[j, b] * temp1[a, j, :, :]
            for c in range(num_modes):
                for k in range(num_modes):
                    temp3[a, b, c, :] += unitary_matrix_dagger[k, c] * temp2[a, b, k, :]
                for d in range(num_modes):
                    for l in range(num_modes):
                        temp_ret[a, b, c, d] += unitary_matrix[l, d] * temp3[a, b, c, l]
    return temp_ret


class TestFermionicOperator(QiskitChemistryTestCase):
    """Fermionic Operator tests."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.595',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        molecule = driver.run()
        self.fer_op = FermionicOperator(h1=molecule.one_body_integrals,
                                        h2=molecule.two_body_integrals)

    def test_transform(self):
        unitary_matrix = random_unitary(self.fer_op.h1.shape[0])

        reference_fer_op = copy.deepcopy(self.fer_op)
        target_fer_op = copy.deepcopy(self.fer_op)

        reference_fer_op._h1_transform(unitary_matrix)
        reference_fer_op.h2 = h2_transform_slow(reference_fer_op.h2, unitary_matrix)

        target_fer_op._h1_transform(unitary_matrix)
        target_fer_op._h2_transform(unitary_matrix)

        h1_nonzeros = np.count_nonzero(reference_fer_op.h1 - target_fer_op.h1)
        self.assertEqual(h1_nonzeros, 0, "there are differences between h1 transformation")

        h2_nonzeros = np.count_nonzero(reference_fer_op.h2 - target_fer_op.h2)
        self.assertEqual(h2_nonzeros, 0, "there are differences between h2 transformation")

    def test_freezing_core(self):
        driver = PySCFDriver(atom='H .0 .0 -1.160518; Li .0 .0 0.386839',
                             unit=UnitsType.ANGSTROM,
                             charge=0,
                             spin=0,
                             basis='sto3g')
        molecule = driver.run()
        fer_op = FermionicOperator(h1=molecule.one_body_integrals,
                                   h2=molecule.two_body_integrals)
        fer_op, energy_shift = fer_op.fermion_mode_freezing([0, 6])
        gt = -7.8187092970493755
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

        driver = PySCFDriver(atom='H .0 .0 .0; Na .0 .0 1.888',
                             unit=UnitsType.ANGSTROM,
                             charge=0,
                             spin=0,
                             basis='sto3g')
        molecule = driver.run()
        fer_op = FermionicOperator(h1=molecule.one_body_integrals,
                                   h2=molecule.two_body_integrals)
        fer_op, energy_shift = fer_op.fermion_mode_freezing([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        gt = -162.58414559586748
        diff = abs(energy_shift - gt)
        self.assertLess(diff, 1e-6)

    def test_bksf_mapping(self):
        """Test bksf mapping.

        The spectrum of bksf mapping should be half of jordan wigner mapping.
        """
        driver = PySCFDriver(atom='H .0 .0 0.7414; H .0 .0 .0',
                             unit=UnitsType.ANGSTROM,
                             charge=0,
                             spin=0,
                             basis='sto3g')
        molecule = driver.run()
        fer_op = FermionicOperator(h1=molecule.one_body_integrals,
                                   h2=molecule.two_body_integrals)
        jw_op = fer_op.mapping('jordan_wigner')
        bksf_op = fer_op.mapping('bksf')
        jw_op.to_matrix()
        bksf_op.to_matrix()
        jw_eigs = np.linalg.eigvals(jw_op.matrix.toarray())
        bksf_eigs = np.linalg.eigvals(bksf_op.matrix.toarray())

        jw_eigs = np.sort(np.around(jw_eigs.real, 6))
        bksf_eigs = np.sort(np.around(bksf_eigs.real, 6))
        overlapped_spectrum = np.sum(np.isin(jw_eigs, bksf_eigs))

        self.assertEqual(overlapped_spectrum, jw_eigs.size // 2)


if __name__ == '__main__':
    unittest.main()

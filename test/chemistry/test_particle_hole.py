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

""" Test Particle Hole """

from test.chemistry import QiskitChemistryTestCase
from ddt import ddt, idata, unpack
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.chemistry import FermionicOperator, QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType


@ddt
class TestParticleHole(QiskitChemistryTestCase):
    """Test ParticleHole transformations of Fermionic Operator"""

    H_2 = 'H 0 0 0; H 0 0 0.735'
    LIH = 'Li 0 0 0; H 0 0 1.6'
    H2O = 'H; O 1 1.08; H 2 1.08 1 107.5'
    O_H = 'O 0 0 0; H 0 0 0.9697'
    CH2 = 'C; H 1 1; H 1 1 2 125.0'

    @idata([
        [H_2, 0, 0, 'sto3g', HFMethodType.RHF],
        [H_2, 0, 0, '6-31g', HFMethodType.RHF],
        [LIH, 0, 0, 'sto3g', HFMethodType.RHF],
        [LIH, 0, 0, 'sto3g', HFMethodType.ROHF],
        [LIH, 0, 0, 'sto3g', HFMethodType.UHF],
        [H2O, 0, 0, 'sto3g', HFMethodType.RHF],
        [O_H, 0, 1, 'sto3g', HFMethodType.ROHF],
        [O_H, 0, 1, 'sto3g', HFMethodType.UHF],
        [CH2, 0, 2, 'sto3g', HFMethodType.ROHF],
        [CH2, 0, 2, 'sto3g', HFMethodType.UHF],
    ])
    @unpack
    def test_particle_hole(self, atom, charge=0, spin=0, basis='sto3g', hf_method=HFMethodType.RHF):
        """ particle hole test """
        try:
            driver = PySCFDriver(atom=atom,
                                 unit=UnitsType.ANGSTROM,
                                 charge=charge,
                                 spin=spin,
                                 basis=basis,
                                 hf_method=hf_method)
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        config = '{}, charge={}, spin={}, basis={}, {}'.format(atom, charge,
                                                               spin, basis,
                                                               hf_method.value)

        molecule = driver.run()
        fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

        ph_fer_op, ph_shift = fer_op.particle_hole_transformation([molecule.num_alpha,
                                                                   molecule.num_beta])

        # ph_shift should be the electronic part of the hartree fock energy
        self.assertAlmostEqual(-ph_shift,
                               molecule.hf_energy - molecule.nuclear_repulsion_energy, msg=config)

        # Energy in original fer_op should same as ph transformed one added with ph_shift
        jw_op = fer_op.mapping('jordan_wigner')
        result = NumPyMinimumEigensolver(jw_op).run()

        ph_jw_op = ph_fer_op.mapping('jordan_wigner')
        ph_result = NumPyMinimumEigensolver(ph_jw_op).run()

        self.assertAlmostEqual(result.eigenvalue.real,
                               ph_result.eigenvalue.real - ph_shift, msg=config)

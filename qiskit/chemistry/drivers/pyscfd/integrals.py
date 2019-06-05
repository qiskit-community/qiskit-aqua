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

import logging
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import QMolecule
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pyscf import gto, scf, ao2mo
    from pyscf import __version__ as pyscf_version
    from pyscf.lib import param
    from pyscf.lib import logger as pylogger
except ImportError:
    logger.info("PySCF is not installed. Use 'pip install pyscf'")


def compute_integrals(atom,
                      unit,
                      charge,
                      spin,
                      basis,
                      hf_method='rhf',
                      max_memory=None):
    # Get config from input parameters
    # molecule is in PySCF atom string format e.g. "H .0 .0 .0; H .0 .0 0.2"
    #          or in Z-Matrix format e.g. "H; O 1 1.08; H 2 1.08 1 107.5"
    # other parameters are as per PySCF got.Mole format

    atom = _check_molecule_format(atom)
    hf_method = hf_method.lower()
    if max_memory is None:
        max_memory = param.MAX_MEMORY

    try:
        mol = gto.Mole(atom=atom, unit=unit, basis=basis, max_memory=max_memory, verbose=pylogger.QUIET)
        mol.symmetry = False
        mol.charge = charge
        mol.spin = spin
        mol.build(parse_arg=False)
        q_mol = _calculate_integrals(mol, hf_method)
    except Exception as exc:
        raise QiskitChemistryError('Failed electronic structure computation') from exc

    return q_mol


def _check_molecule_format(val):
    """If it seems to be zmatrix rather than xyz format we convert before returning"""
    atoms = [x.strip() for x in val.split(';')]
    if atoms is None or len(atoms) < 1:
        raise QiskitChemistryError('Molecule format error: ' + val)

    # An xyz format has 4 parts in each atom, if not then do zmatrix convert
    # Allows dummy atoms, using symbol 'X' in zmatrix format for coord computation to xyz
    parts = [x.strip() for x in atoms[0].split(' ')]
    if len(parts) != 4:
        try:
            newval = []
            for entry in gto.mole.from_zmatrix(val):
                if entry[0].upper() != 'X':
                    newval.append(entry)
            return newval
        except Exception as exc:
            raise QiskitChemistryError('Failed to convert atom string: ' + val) from exc

    return val


def _calculate_integrals(mol, hf_method='rhf'):
    """Function to calculate the one and two electron terms. Perform a Hartree-Fock calculation in
        the given basis.
    Args:
        mol (gto.Mole) : A PySCF gto.Mole object.
        hf_method (str): rhf, uhf, rohf
    Returns:
        QMolecule: QMolecule populated with driver integrals etc
    """
    enuke = gto.mole.energy_nuc(mol)

    if hf_method == 'rhf':
        mf = scf.RHF(mol)
    elif hf_method == 'rohf':
        mf = scf.ROHF(mol)
    elif hf_method == 'uhf':
        mf = scf.UHF(mol)
    else:
        raise QiskitChemistryError('Invalid hf_method type: {}'.format(hf_method))

    ehf = mf.kernel()
    if type(mf.mo_coeff) is tuple:
        mo_coeff = mf.mo_coeff[0]
        mo_coeff_B = mf.mo_coeff[1]
        # mo_occ   = mf.mo_occ[0]
        # mo_occ_B = mf.mo_occ[1]
    else:
        mo_coeff = mf.mo_coeff
        mo_coeff_B = None
        # mo_occ   = mf.mo_occ
        # mo_occ_B = None
    norbs = mo_coeff.shape[0]

    if type(mf.mo_energy) is tuple:
        orbs_energy = mf.mo_energy[0]
        orbs_energy_B = mf.mo_energy[1]
    else:
        orbs_energy = mf.mo_energy
        orbs_energy_B = None

    hij = mf.get_hcore()
    mohij = np.dot(np.dot(mo_coeff.T, hij), mo_coeff)
    mohij_B = None
    if mo_coeff_B is not None:
        mohij_B = np.dot(np.dot(mo_coeff_B.T, hij), mo_coeff_B)

    eri = mol.intor('int2e', aosym=1)
    mo_eri = ao2mo.incore.full(mf._eri, mo_coeff, compact=False)
    mohijkl = mo_eri.reshape(norbs, norbs, norbs, norbs)
    mohijkl_BB = None
    mohijkl_BA = None
    if mo_coeff_B is not None:
        mo_eri_B = ao2mo.incore.full(mf._eri, mo_coeff_B, compact=False)
        mohijkl_BB = mo_eri_B.reshape(norbs, norbs, norbs, norbs)
        mo_eri_BA = ao2mo.incore.general(mf._eri, (mo_coeff_B, mo_coeff_B, mo_coeff, mo_coeff), compact=False)
        mohijkl_BA = mo_eri_BA.reshape(norbs, norbs, norbs, norbs)

    # dipole integrals
    mol.set_common_orig((0, 0, 0))
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    x_dip_ints = ao_dip[0]
    y_dip_ints = ao_dip[1]
    z_dip_ints = ao_dip[2]

    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    if hf_method == 'rohf' or hf_method == 'uhf':
        dm = dm[0]
    elec_dip = np.negative(np.einsum('xij,ji->x', ao_dip, dm).real)
    elec_dip = np.round(elec_dip, decimals=8)
    nucl_dip = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    nucl_dip = np.round(nucl_dip, decimals=8)
    logger.info("HF Electronic dipole moment: {}".format(elec_dip))
    logger.info("Nuclear dipole moment: {}".format(nucl_dip))
    logger.info("Total dipole moment: {}".format(nucl_dip+elec_dip))

    # Create driver level molecule object and populate
    _q_ = QMolecule()
    _q_.origin_driver_version = pyscf_version
    # Energies and orbits
    _q_.hf_energy = ehf
    _q_.nuclear_repulsion_energy = enuke
    _q_.num_orbitals = norbs
    _q_.num_alpha = mol.nelec[0]
    _q_.num_beta = mol.nelec[1]
    _q_.mo_coeff = mo_coeff
    _q_.mo_coeff_B = mo_coeff_B
    _q_.orbital_energies = orbs_energy
    _q_.orbital_energies_B = orbs_energy_B
    # Molecule geometry
    _q_.molecular_charge = mol.charge
    _q_.multiplicity = mol.spin + 1
    _q_.num_atoms = mol.natm
    _q_.atom_symbol = []
    _q_.atom_xyz = np.empty([mol.natm, 3])
    atoms = mol.atom_coords()
    for _n in range(0, _q_.num_atoms):
        xyz = mol.atom_coord(_n)
        _q_.atom_symbol.append(mol.atom_pure_symbol(_n))
        _q_.atom_xyz[_n][0] = xyz[0]
        _q_.atom_xyz[_n][1] = xyz[1]
        _q_.atom_xyz[_n][2] = xyz[2]
    # 1 and 2 electron integrals AO and MO
    _q_.hcore = hij
    _q_.hcore_B = None
    _q_.kinetic = mol.intor_symmetric('int1e_kin')
    _q_.overlap = mf.get_ovlp()
    _q_.eri = eri
    _q_.mo_onee_ints = mohij
    _q_.mo_onee_ints_B = mohij_B
    _q_.mo_eri_ints = mohijkl
    _q_.mo_eri_ints_BB = mohijkl_BB
    _q_.mo_eri_ints_BA = mohijkl_BA
    # dipole integrals AO and MO
    _q_.x_dip_ints = x_dip_ints
    _q_.y_dip_ints = y_dip_ints
    _q_.z_dip_ints = z_dip_ints
    _q_.x_dip_mo_ints = QMolecule.oneeints2mo(x_dip_ints, mo_coeff)
    _q_.x_dip_mo_ints_B = None
    _q_.y_dip_mo_ints = QMolecule.oneeints2mo(y_dip_ints, mo_coeff)
    _q_.y_dip_mo_ints_B = None
    _q_.z_dip_mo_ints = QMolecule.oneeints2mo(z_dip_ints, mo_coeff)
    _q_.z_dip_mo_ints_B = None
    if mo_coeff_B is not None:
        _q_.x_dip_mo_ints_B = QMolecule.oneeints2mo(x_dip_ints, mo_coeff_B)
        _q_.y_dip_mo_ints_B = QMolecule.oneeints2mo(y_dip_ints, mo_coeff_B)
        _q_.z_dip_mo_ints_B = QMolecule.oneeints2mo(z_dip_ints, mo_coeff_B)
    # dipole moment
    _q_.nuclear_dipole_moment = nucl_dip
    _q_.reverse_dipole_sign = True

    return _q_

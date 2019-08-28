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

""" Integrals methods """

import logging
import tempfile
import os
import numpy as np
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import QMolecule

logger = logging.getLogger(__name__)

try:
    from pyscf import gto, scf, ao2mo
    from pyscf import __version__ as pyscf_version
    from pyscf.lib import param
    from pyscf.lib import logger as pylogger
    from pyscf.tools import dump_mat
except ImportError:
    logger.info("PySCF is not installed. See https://sunqm.github.io/pyscf/install.html")


def compute_integrals(atom,
                      unit,
                      charge,
                      spin,
                      basis,
                      hf_method='rhf',
                      conv_tol=1e-9,
                      max_cycle=50,
                      init_guess='minao',
                      max_memory=None):
    """ compute integrals """
    # Get config from input parameters
    # molecule is in PySCF atom string format e.g. "H .0 .0 .0; H .0 .0 0.2"
    #          or in Z-Matrix format e.g. "H; O 1 1.08; H 2 1.08 1 107.5"
    # other parameters are as per PySCF got.Mole format

    atom = _check_molecule_format(atom)
    hf_method = hf_method.lower()
    if max_memory is None:
        max_memory = param.MAX_MEMORY

    try:
        verbose = pylogger.QUIET
        output = None
        if logger.isEnabledFor(logging.DEBUG):
            verbose = pylogger.INFO
            file, output = tempfile.mkstemp(suffix='.log')
            os.close(file)

        mol = gto.Mole(atom=atom, unit=unit, basis=basis,
                       max_memory=max_memory, verbose=verbose, output=output)
        mol.symmetry = False
        mol.charge = charge
        mol.spin = spin
        mol.build(parse_arg=False)
        q_mol = _calculate_integrals(mol, hf_method, conv_tol, max_cycle, init_guess)
        if output is not None:
            _process_pyscf_log(output)
            try:
                os.remove(output)
            except Exception:  # pylint: disable=broad-except
                pass

    except Exception as exc:
        raise QiskitChemistryError('Failed electronic structure computation') from exc

    return q_mol


def _check_molecule_format(val):
    """If it seems to be zmatrix rather than xyz format we convert before returning"""
    atoms = [x.strip() for x in val.split(';')]
    if atoms is None or len(atoms) < 1:  # pylint: disable=len-as-condition
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


def _calculate_integrals(mol, hf_method='rhf', conv_tol=1e-9, max_cycle=50, init_guess='minao'):
    """Function to calculate the one and two electron terms. Perform a Hartree-Fock calculation in
        the given basis.
    Args:
        mol (gto.Mole) : A PySCF gto.Mole object.
        hf_method (str): rhf, uhf, rohf
        conv_tol (float): Convergence tolerance
        max_cycle (int): Max convergence cycles
        init_guess (str): Initial guess for SCF
    Returns:
        QMolecule: QMolecule populated with driver integrals etc
    Raises:
        QiskitChemistryError: Invalid hf method type
    """
    enuke = gto.mole.energy_nuc(mol)

    if hf_method == 'rhf':
        m_f = scf.RHF(mol)
    elif hf_method == 'rohf':
        m_f = scf.ROHF(mol)
    elif hf_method == 'uhf':
        m_f = scf.UHF(mol)
    else:
        raise QiskitChemistryError('Invalid hf_method type: {}'.format(hf_method))

    m_f.conv_tol = conv_tol
    m_f.max_cycle = max_cycle
    m_f.init_guess = init_guess
    ehf = m_f.kernel()
    logger.info('PySCF kernel() converged: %s, e(hf): %s', m_f.converged, m_f.e_tot)
    if isinstance(m_f.mo_coeff, tuple):
        mo_coeff = m_f.mo_coeff[0]
        mo_coeff_b = m_f.mo_coeff[1]
        # mo_occ   = m_f.mo_occ[0]
        # mo_occ_b = m_f.mo_occ[1]
    else:
        # With PySCF 1.6.2, instead of a tuple of 2 dimensional arrays, its a 3 dimensional
        # array with the first dimension indexing to the coeff arrays for alpha and beta
        if len(m_f.mo_coeff.shape) > 2:
            mo_coeff = m_f.mo_coeff[0]
            mo_coeff_b = m_f.mo_coeff[1]
            # mo_occ   = m_f.mo_occ[0]
            # mo_occ_b = m_f.mo_occ[1]
        else:
            mo_coeff = m_f.mo_coeff
            mo_coeff_b = None
            # mo_occ   = mf.mo_occ
            # mo_occ_b = None
    norbs = mo_coeff.shape[0]

    if isinstance(m_f.mo_energy, tuple):
        orbs_energy = m_f.mo_energy[0]
        orbs_energy_b = m_f.mo_energy[1]
    else:
        # See PYSCF 1.6.2 comment above - this was similarly changed
        if len(m_f.mo_energy.shape) > 1:
            orbs_energy = m_f.mo_energy[0]
            orbs_energy_b = m_f.mo_energy[1]
        else:
            orbs_energy = m_f.mo_energy
            orbs_energy_b = None

    if logger.isEnabledFor(logging.DEBUG):
        # Add some more to PySCF output...
        # First analyze() which prints extra information about MO energy and occupation
        mol.stdout.write('\n')
        m_f.analyze()
        # Now labelled orbitals for contributions to the MOs for s,p,d etc of each atom
        mol.stdout.write('\n\n--- Alpha Molecular Orbitals ---\n\n')
        dump_mat.dump_mo(mol, mo_coeff, digits=7, start=1)
        if mo_coeff_b is not None:
            mol.stdout.write('\n--- Beta Molecular Orbitals ---\n\n')
            dump_mat.dump_mo(mol, mo_coeff_b, digits=7, start=1)
        mol.stdout.flush()

    hij = m_f.get_hcore()
    mohij = np.dot(np.dot(mo_coeff.T, hij), mo_coeff)
    mohij_b = None
    if mo_coeff_b is not None:
        mohij_b = np.dot(np.dot(mo_coeff_b.T, hij), mo_coeff_b)

    eri = mol.intor('int2e', aosym=1)
    mo_eri = ao2mo.incore.full(m_f._eri, mo_coeff, compact=False)
    mohijkl = mo_eri.reshape(norbs, norbs, norbs, norbs)
    mohijkl_bb = None
    mohijkl_ba = None
    if mo_coeff_b is not None:
        mo_eri_b = ao2mo.incore.full(m_f._eri, mo_coeff_b, compact=False)
        mohijkl_bb = mo_eri_b.reshape(norbs, norbs, norbs, norbs)
        mo_eri_ba = ao2mo.incore.general(m_f._eri,
                                         (mo_coeff_b, mo_coeff_b, mo_coeff, mo_coeff),
                                         compact=False)
        mohijkl_ba = mo_eri_ba.reshape(norbs, norbs, norbs, norbs)

    # dipole integrals
    mol.set_common_orig((0, 0, 0))
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    x_dip_ints = ao_dip[0]
    y_dip_ints = ao_dip[1]
    z_dip_ints = ao_dip[2]

    d_m = m_f.make_rdm1(m_f.mo_coeff, m_f.mo_occ)
    if hf_method in ('rohf', 'uhf'):
        d_m = d_m[0]
    elec_dip = np.negative(np.einsum('xij,ji->x', ao_dip, d_m).real)
    elec_dip = np.round(elec_dip, decimals=8)
    nucl_dip = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    nucl_dip = np.round(nucl_dip, decimals=8)
    logger.info("HF Electronic dipole moment: %s", elec_dip)
    logger.info("Nuclear dipole moment: %s", nucl_dip)
    logger.info("Total dipole moment: %s", nucl_dip+elec_dip)

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
    _q_.mo_coeff_b = mo_coeff_b
    _q_.orbital_energies = orbs_energy
    _q_.orbital_energies_b = orbs_energy_b
    # Molecule geometry
    _q_.molecular_charge = mol.charge
    _q_.multiplicity = mol.spin + 1
    _q_.num_atoms = mol.natm
    _q_.atom_symbol = []
    _q_.atom_xyz = np.empty([mol.natm, 3])
    _ = mol.atom_coords()
    for n_i in range(0, _q_.num_atoms):
        xyz = mol.atom_coord(n_i)
        _q_.atom_symbol.append(mol.atom_pure_symbol(n_i))
        _q_.atom_xyz[n_i][0] = xyz[0]
        _q_.atom_xyz[n_i][1] = xyz[1]
        _q_.atom_xyz[n_i][2] = xyz[2]
    # 1 and 2 electron integrals AO and MO
    _q_.hcore = hij
    _q_.hcore_b = None
    _q_.kinetic = mol.intor_symmetric('int1e_kin')
    _q_.overlap = m_f.get_ovlp()
    _q_.eri = eri
    _q_.mo_onee_ints = mohij
    _q_.mo_onee_ints_b = mohij_b
    _q_.mo_eri_ints = mohijkl
    _q_.mo_eri_ints_bb = mohijkl_bb
    _q_.mo_eri_ints_ba = mohijkl_ba
    # dipole integrals AO and MO
    _q_.x_dip_ints = x_dip_ints
    _q_.y_dip_ints = y_dip_ints
    _q_.z_dip_ints = z_dip_ints
    _q_.x_dip_mo_ints = QMolecule.oneeints2mo(x_dip_ints, mo_coeff)
    _q_.x_dip_mo_ints_b = None
    _q_.y_dip_mo_ints = QMolecule.oneeints2mo(y_dip_ints, mo_coeff)
    _q_.y_dip_mo_ints_b = None
    _q_.z_dip_mo_ints = QMolecule.oneeints2mo(z_dip_ints, mo_coeff)
    _q_.z_dip_mo_ints_b = None
    if mo_coeff_b is not None:
        _q_.x_dip_mo_ints_b = QMolecule.oneeints2mo(x_dip_ints, mo_coeff_b)
        _q_.y_dip_mo_ints_b = QMolecule.oneeints2mo(y_dip_ints, mo_coeff_b)
        _q_.z_dip_mo_ints_b = QMolecule.oneeints2mo(z_dip_ints, mo_coeff_b)
    # dipole moment
    _q_.nuclear_dipole_moment = nucl_dip
    _q_.reverse_dipole_sign = True

    return _q_


def _process_pyscf_log(logfile):
    with open(logfile) as file:
        content = file.readlines()

    for i, _ in enumerate(content):
        if content[i].startswith('System:'):
            content = content[i:]
            break

    logger.debug('PySCF processing messages log:\n%s', ''.join(content))

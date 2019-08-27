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

import re
import logging
import numpy as np
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import QMolecule

logger = logging.getLogger(__name__)

try:
    from pyquante2 import rhf, uhf, rohf, basisset, onee_integrals
    from pyquante2.geo.zmatrix import z2xyz
    from pyquante2.ints.integrals import twoe_integrals
    from pyquante2.utils import simx
except ImportError:
    logger.info('PyQuante2 is not installed. See https://github.com/rpmuller/pyquante2')


def compute_integrals(atoms,
                      units,
                      charge,
                      multiplicity,
                      basis,
                      hf_method='rhf',
                      tol=1e-8,
                      maxiters=100):
    """ Compute integrals """
    # Get config from input parameters
    # Molecule is in this format xyz as below or in Z-matrix e.g "H; O 1 1.08; H 2 1.08 1 107.5":
    # atoms=H .0 .0 .0; H .0 .0 0.2
    # units=Angstrom
    # charge=0
    # multiplicity=1
    # where we support symbol for atom as well as number

    units = _check_units(units)
    mol = _parse_molecule(atoms, units, charge, multiplicity)
    hf_method = hf_method.lower()

    try:
        q_mol = _calculate_integrals(mol, basis, hf_method, tol, maxiters)
    except Exception as exc:
        raise QiskitChemistryError('Failed electronic structure computation') from exc

    return q_mol


def _calculate_integrals(molecule, basis='sto3g', hf_method='rhf', tol=1e-8, maxiters=100):
    """Function to calculate the one and two electron terms. Perform a Hartree-Fock calculation in
        the given basis.
    Args:
        molecule (pyQuante2.molecule): A pyquante2 molecular object.
        basis (str) : The basis set for the electronic structure computation
        hf_method (str): rhf, uhf, rohf
        tol (float): tolerance
        maxiters (int): max. iterations
    Returns:
        QMolecule: QMolecule populated with driver integrals etc
    Raises:
        QiskitChemistryError: Invalid hf methos type
    """
    bfs = basisset(molecule, basis)
    integrals = onee_integrals(bfs, molecule)
    hij = integrals.T + integrals.V
    hijkl = twoe_integrals(bfs)

    # convert overlap integrals to molecular basis
    # calculate the Hartree-Fock solution of the molecule

    if hf_method == 'rhf':
        solver = rhf(molecule, bfs)
    elif hf_method == 'rohf':
        solver = rohf(molecule, bfs)
    elif hf_method == 'uhf':
        solver = uhf(molecule, bfs)
    else:
        raise QiskitChemistryError('Invalid hf_method type: {}'.format(hf_method))
    ehf = solver.converge(tol=tol, maxiters=maxiters)
    logger.debug('PyQuante2 processing information:\n%s', solver)
    if hasattr(solver, 'orbs'):
        orbs = solver.orbs
        orbs_b = None
    else:
        orbs = solver.orbsa
        orbs_b = solver.orbsb
    norbs = len(orbs)
    if hasattr(solver, 'orbe'):
        orbs_energy = solver.orbe
        orbs_energy_b = None
    else:
        orbs_energy = solver.orbea
        orbs_energy_b = solver.orbeb
    enuke = molecule.nuclear_repulsion()
    # Get ints in molecular orbital basis
    mohij = simx(hij, orbs)
    mohij_b = None
    if orbs_b is not None:
        mohij_b = simx(hij, orbs_b)

    eri = hijkl.transform(np.identity(norbs))
    mohijkl = hijkl.transform(orbs)
    mohijkl_bb = None
    mohijkl_ba = None
    if orbs_b is not None:
        mohijkl_bb = hijkl.transform(orbs_b)
        mohijkl_ba = np.einsum('aI,bJ,cK,dL,abcd->IJKL', orbs_b, orbs_b, orbs, orbs, hijkl[...])

    # Create driver level molecule object and populate
    _q_ = QMolecule()
    _q_.origin_driver_version = '?'  # No version info seems available to access
    # Energies and orbits
    _q_.hf_energy = ehf[0]
    _q_.nuclear_repulsion_energy = enuke
    _q_.num_orbitals = norbs
    _q_.num_alpha = molecule.nup()
    _q_.num_beta = molecule.ndown()
    _q_.mo_coeff = orbs
    _q_.mo_coeff_b = orbs_b
    _q_.orbital_energies = orbs_energy
    _q_.orbital_energies_b = orbs_energy_b
    # Molecule geometry
    _q_.molecular_charge = molecule.charge
    _q_.multiplicity = molecule.multiplicity
    _q_.num_atoms = len(molecule)
    _q_.atom_symbol = []
    _q_.atom_xyz = np.empty([len(molecule), 3])
    atoms = molecule.atoms
    for n_i in range(0, _q_.num_atoms):
        atuple = atoms[n_i].atuple()
        _q_.atom_symbol.append(QMolecule.symbols[atuple[0]])
        _q_.atom_xyz[n_i][0] = atuple[1]
        _q_.atom_xyz[n_i][1] = atuple[2]
        _q_.atom_xyz[n_i][2] = atuple[3]
    # 1 and 2 electron integrals
    _q_.hcore = hij
    _q_.hcore_b = None
    _q_.kinetic = integrals.T
    _q_.overlap = integrals.S
    _q_.eri = eri
    _q_.mo_onee_ints = mohij
    _q_.mo_onee_ints_b = mohij_b
    _q_.mo_eri_ints = mohijkl
    _q_.mo_eri_ints_bb = mohijkl_bb
    _q_.mo_eri_ints_ba = mohijkl_ba

    return _q_


def _parse_molecule(val, units, charge, multiplicity):
    val = _check_molecule_format(val)

    parts = [x.strip() for x in val.split(';')]
    if parts is None or len(parts) < 1:  # pylint: disable=len-as-condition
        raise QiskitChemistryError('Molecule format error: ' + val)
    geom = []
    for n, _ in enumerate(parts):
        part = parts[n]
        geom.append(_parse_atom(part))

    if len(geom) < 1:  # pylint: disable=len-as-condition
        raise QiskitChemistryError('Molecule format error: ' + val)

    try:
        from pyquante2 import molecule
        return molecule(geom, units=units, charge=charge, multiplicity=multiplicity)
    except Exception as exc:
        raise QiskitChemistryError('Failed to create molecule') from exc


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
            zmat = []
            for atom in atoms:
                parts = [x.strip() for x in atom.split(' ')]
                z = [parts[0]]
                for i in range(1, len(parts), 2):
                    z.append(int(parts[i]))
                    z.append(float(parts[i+1]))
                zmat.append(z)
            xyz = z2xyz(zmat)
            new_val = ""
            for atm in xyz:
                if atm[0].upper() == 'X':
                    continue
                if new_val:
                    new_val += "; "
                new_val += "{} {} {} {}".format(atm[0], atm[1], atm[2], atm[3])
            return new_val
        except Exception as exc:
            raise QiskitChemistryError('Failed to convert atom string: ' + val) from exc

    return val


def _parse_atom(val):
    if val is None or len(val) < 1:  # pylint: disable=len-as-condition
        raise QiskitChemistryError('Molecule atom format error: empty')

    parts = re.split(r'\s+', val)
    if len(parts) != 4:
        raise QiskitChemistryError('Molecule atom format error: ' + val)

    parts[0] = parts[0].lower().capitalize()
    if not parts[0].isdigit():
        if parts[0] in QMolecule.symbols:
            parts[0] = QMolecule.symbols.index(parts[0])
        else:
            raise QiskitChemistryError('Molecule atom symbol error: ' + parts[0])

    return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3])


def _check_units(units):
    if units.lower() in ["angstrom", "ang", "a"]:
        units = 'Angstrom'
    elif units.lower() in ["bohr", "b"]:
        units = 'Bohr'
    else:
        raise QiskitChemistryError('Molecule units format error: ' + units)
    return units

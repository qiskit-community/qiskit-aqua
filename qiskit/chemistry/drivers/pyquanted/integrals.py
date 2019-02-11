# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================#

from .transform import transformintegrals, ijkl2intindex
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import QMolecule
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

try:
    from pyquante2 import molecule, rhf, uhf, rohf, basisset, onee_integrals
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
                      calc_type='rhf'):
    # Get config from input parameters
    # Molecule is in this format xyz as below or in Z-matrix e.g "H; O 1 1.08; H 2 1.08 1 107.5":
    # atoms=H .0 .0 .0; H .0 .0 0.2
    # units=Angstrom
    # charge=0
    # multiplicity=1
    # where we support symbol for atom as well as number

    units = _check_units(units)
    mol = _parse_molecule(atoms, units, charge, multiplicity)
    calc_type = calc_type.lower()

    try:
        ehf, enuke, norbs, mohij, mohijkl, orbs, orbs_energy = _calculate_integrals(mol, basis, calc_type)
    except Exception as exc:
        raise QiskitChemistryError('Failed electronic structure computation') from exc

    # Create driver level molecule object and populate
    _q_ = QMolecule()
    # Energies and orbits
    _q_.hf_energy = ehf
    _q_.nuclear_repulsion_energy = enuke
    _q_.num_orbitals = norbs
    _q_.num_alpha = mol.nup()
    _q_.num_beta = mol.ndown()
    _q_.mo_coeff = orbs
    _q_.orbital_energies = orbs_energy
    # Molecule geometry
    _q_.molecular_charge = mol.charge
    _q_.multiplicity = mol.multiplicity
    _q_.num_atoms = len(mol)
    _q_.atom_symbol = []
    _q_.atom_xyz = np.empty([len(mol), 3])
    atoms = mol.atoms
    for _n in range(0, _q_.num_atoms):
        atuple = atoms[_n].atuple()
        _q_.atom_symbol.append(QMolecule.symbols[atuple[0]])
        _q_.atom_xyz[_n][0] = atuple[1]
        _q_.atom_xyz[_n][1] = atuple[2]
        _q_.atom_xyz[_n][2] = atuple[3]
    # 1 and 2 electron integrals
    _q_.mo_onee_ints = mohij
    _q_.mo_eri_ints = mohijkl

    return _q_


def _calculate_integrals(molecule, basis='sto3g', calc_type='rhf'):
    """Function to calculate the one and two electron terms. Perform a Hartree-Fock calculation in
        the given basis.
    Args:
        molecule : A pyquante2 molecular object.
        basis : The basis set for the electronic structure computation
        calc_type: rhf, uhf, rohf
    Returns:
        ehf : Hartree-Fock energy
        enuke: Nuclear repulsion energy
        norbs : Number of orbitals
        mohij : One electron terms of the Hamiltonian.
        mohijkl : Two electron terms of the Hamiltonian.
        orbs: Molecular orbital coefficients
        orbs_energy: Orbital energies
    """
    bfs = basisset(molecule, basis)
    integrals = onee_integrals(bfs, molecule)
    hij = integrals.T + integrals.V
    hijkl_compressed = twoe_integrals(bfs)

    # convert overlap integrals to molecular basis
    # calculate the Hartree-Fock solution of the molecule

    if calc_type == 'rhf':
        solver = rhf(molecule, bfs)
    elif calc_type == 'rohf':
        solver = rohf(molecule, bfs)
    elif calc_type == 'uhf':
        solver = uhf(molecule, bfs)
    else:
        raise QiskitChemistryError('Invalid calc_type: {}'.format(calc_type))
    logger.debug('Solver name {}'.format(solver.name))
    ehf = solver.converge()
    if hasattr(solver, 'orbs'):
        orbs = solver.orbs
    else:
        orbs = solver.orbsa
    norbs = len(orbs)
    if hasattr(solver, 'orbe'):
        orbs_energy = solver.orbe
    else:
        orbs_energy = solver.orbea
    enuke = molecule.nuclear_repulsion()
    # Get ints in molecular orbital basis
    mohij = simx(hij, orbs)
    mohijkl_compressed = transformintegrals(hijkl_compressed, orbs)
    mohijkl = np.zeros((norbs, norbs, norbs, norbs))
    for i in range(norbs):
        for j in range(norbs):
            for k in range(norbs):
                for l in range(norbs):
                    mohijkl[i, j, k, l] = mohijkl_compressed[ijkl2intindex(i, j, k, l)]

    return ehf[0], enuke, norbs, mohij, mohijkl, orbs, orbs_energy


def _parse_molecule(val, units, charge, multiplicity):
    val = _check_molecule_format(val)

    parts = [x.strip() for x in val.split(';')]
    if parts is None or len(parts) < 1:
        raise QiskitChemistryError('Molecule format error: ' + val)
    geom = []
    for n in range(len(parts)):
        part = parts[n]
        geom.append(_parse_atom(part))

    if len(geom) < 1:
        raise QiskitChemistryError('Molecule format error: ' + val)

    try:
        return molecule(geom, units=units, charge=charge, multiplicity=multiplicity)
    except Exception as exc:
        raise QiskitChemistryError('Failed to create molecule') from exc


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
            for i in range(len(xyz)):
                atm = xyz[i]
                if atm[0].upper() == 'X':
                    continue
                if len(new_val) > 0:
                    new_val += "; "
                new_val += "{} {} {} {}".format(atm[0], atm[1], atm[2], atm[3])
            return new_val
        except Exception as exc:
            raise QiskitChemistryError('Failed to convert atom string: ' + val) from exc

    return val


def _parse_atom(val):
    if val is None or len(val) < 1:
        raise QiskitChemistryError('Molecule atom format error: empty')

    parts = re.split('\s+', val)
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

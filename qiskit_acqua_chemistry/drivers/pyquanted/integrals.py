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

from pyquante2 import molecule, rhf, uhf, rohf, basisset
from pyquante2 import onee_integrals
from pyquante2.ints.integrals import twoe_integrals
from pyquante2.utils import simx
from .transform import transformintegrals, ijkl2intindex
from qiskit_acqua_chemistry import QISChemError
from qiskit_acqua_chemistry import QMolecule
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


def compute_integrals(config):
    # Get config from input parameters
    # Molecule is in this format:
    # atoms=H .0 .0 .0; H .0 .0 0.2
    # units=Angstrom
    # charge=0
    # multiplicity=1
    # where we support symbol for atom as well as number

    if 'atoms' not in config:
        raise QISChemError('Atoms is missing')
    val = config['atoms']
    if val is None:
        raise QISChemError('Atoms value is missing')

    charge = int(config.get('charge', '0'))
    multiplicity = int(config.get('multiplicity', '1'))
    units = __checkUnits(config.get('units', 'Angstrom'))
    mol = __parseMolecule(val, units, charge, multiplicity)
    basis = config.get('basis', 'sto3g')
    calc_type = config.get('calc_type', 'rhf').lower()

    try:
        ehf, enuke, norbs, mohij, mohijkl, orbs, orbs_energy = _calculate_integrals(mol, basis, calc_type)
    except Exception as exc:
        raise QISChemError('Failed electronic structure computation') from exc

    # Create driver level molecule object and populate
    _q_ = QMolecule()
    # Energies and orbits
    _q_._hf_energy = ehf
    _q_._nuclear_repulsion_energy = enuke
    _q_._num_orbitals = norbs
    _q_._num_alpha = mol.nup()
    _q_._num_beta = mol.ndown()
    _q_._mo_coeff = orbs
    _q_._orbital_energies = orbs_energy
    # Molecule geometry
    _q_._molecular_charge = mol.charge
    _q_._multiplicity = mol.multiplicity
    _q_._num_atoms = len(mol)
    _q_._atom_symbol = []
    _q_._atom_xyz = np.empty([len(mol), 3])
    atoms = mol.atoms
    for _n in range(0, _q_._num_atoms):
        atuple = atoms[_n].atuple()
        _q_._atom_symbol.append(QMolecule.symbols[atuple[0]])
        _q_._atom_xyz[_n][0] = atuple[1]
        _q_._atom_xyz[_n][1] = atuple[2]
        _q_._atom_xyz[_n][2] = atuple[3]
    # 1 and 2 electron integrals
    _q_._mo_onee_ints = mohij
    _q_._mo_eri_ints = mohijkl

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
        raise QISChemError('Invalid calc_type: {}'.format(calc_type))
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


def __parseMolecule(val, units, charge, multiplicity):
    parts = [x.strip() for x in val.split(';')]
    if parts is None or len(parts) < 1:
        raise QISChemError('Molecule format error: ' + val)
    geom = []
    for n in range(len(parts)):
        part = parts[n]
        geom.append(__parseAtom(part))

    if len(geom) < 1:
        raise QISChemError('Molecule format error: ' + val)

    try:
        return molecule(geom, units=units, charge=charge, multiplicity=multiplicity)
    except Exception as exc:
        raise QISChemError('Failed to create molecule') from exc


def __parseAtom(val):
    if val is None or len(val) < 1:
        raise QISChemError('Molecule atom format error: ' + val)

    parts = re.split('\s+', val)
    if len(parts) != 4:
        raise QISChemError('Molecule atom format error: ' + val)

    parts[0] = parts[0].lower().capitalize()
    if not parts[0].isdigit():
        if parts[0] in QMolecule.symbols:
            parts[0] = QMolecule.symbols.index(parts[0])
        else:
            raise QISChemError('Molecule atom symbol error: ' + parts[0])

    return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3])


def __checkUnits(units):
    if units.lower() in ["angstrom", "ang", "a"]:
        units = 'Angstrom'
    elif units.lower() in ["bohr", "b"]:
        units = 'Bohr'
    else:
        raise QISChemError('Molecule units format error: ' + units)
    return units

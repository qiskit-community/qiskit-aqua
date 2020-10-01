# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PyQuante Driver """

from typing import Union, List, cast
import importlib
from enum import Enum
import logging
from qiskit.aqua.utils.validation import validate_min
from ..integrals_driver import IntegralsDriver, UnitsType, HFMethodType
from ...qiskit_chemistry_error import QiskitChemistryError
from ...molecule import Molecule
from ...qmolecule import QMolecule
from .integrals import compute_integrals

logger = logging.getLogger(__name__)


class BasisType(Enum):
    """ Basis Type """
    BSTO3G = 'sto3g'
    B631G = '6-31g'
    B631GSS = '6-31g**'


class PyQuanteDriver(IntegralsDriver):
    """
    Qiskit chemistry driver using the PyQuante2 library.

    See https://github.com/rpmuller/pyquante2
    """

    def __init__(self,
                 atoms: Union[str, List[str], Molecule] =
                 'H 0.0 0.0 0.0; H 0.0 0.0 0.735',
                 units: UnitsType = UnitsType.ANGSTROM,
                 charge: int = 0,
                 multiplicity: int = 1,
                 basis: BasisType = BasisType.BSTO3G,
                 hf_method: HFMethodType = HFMethodType.RHF,
                 tol: float = 1e-8,
                 maxiters: int = 100) -> None:
        """
        Args:
            atoms: atoms list or string separated by semicolons or line breaks
            units: angstrom or bohr
            charge: charge
            multiplicity: spin multiplicity
            basis: sto3g or 6-31g or 6-31g**
            hf_method: Hartree-Fock Method type
            tol: Convergence tolerance see pyquante2.scf hamiltonians and iterators
            maxiters: Convergence max iterations see pyquante2.scf hamiltonians and iterators,
                      has a min. value of 1.

        Raises:
            QiskitChemistryError: Invalid Input
        """
        units = units.value
        basis = basis.value
        hf_method = hf_method.value
        validate_min('maxiters', maxiters, 1)
        self._check_valid()
        if not isinstance(atoms, str) and \
                not isinstance(atoms, list) and \
                not isinstance(atoms, Molecule):
            raise QiskitChemistryError("Invalid atom input for PYQUANTE Driver '{}'".format(atoms))

        if isinstance(atoms, list):
            atoms = ';'.join(atoms)
        elif isinstance(atoms, str):
            atoms = atoms.replace('\n', ';')
        elif isinstance(atoms, Molecule):
            if atoms.basis_set is None:
                atoms.basis_set = cast(str, basis)
            if atoms.hf_method is None:
                atoms.hf_method = cast(str, hf_method)

        super().__init__()
        self._atoms = atoms
        self._units = units
        self._charge = charge
        self._multiplicity = multiplicity
        self._basis = cast(str, basis)
        self._hf_method = cast(str, hf_method)
        self._tol = tol
        self._maxiters = maxiters

    @staticmethod
    def _check_valid():
        err_msg = 'PyQuante2 is not installed. See https://github.com/rpmuller/pyquante2'
        try:
            spec = importlib.util.find_spec('pyquante2')
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug('PyQuante2 check error %s', str(ex))
            raise QiskitChemistryError(err_msg) from ex

        raise QiskitChemistryError(err_msg)

    def run(self) -> QMolecule:
        if isinstance(self._atoms, Molecule):
            atoms = ';'.join([name + ' ' + ' '.join(map(str, coord))
                              for (name, coord) in self._atoms.geometry])
            charge = self._atoms.charge
            multiplicity = self._atoms.multiplicity
            basis = self._atoms.basis_set
            hf_method = self._atoms.hf_method
        else:
            atoms = self._atoms
            charge = self._charge
            multiplicity = self._multiplicity
            basis = self._basis
            hf_method = self._hf_method

        q_mol = compute_integrals(atoms=atoms,
                                  units=self._units,
                                  charge=charge,
                                  multiplicity=multiplicity,
                                  basis=basis,
                                  hf_method=hf_method,
                                  tol=self._tol,
                                  maxiters=self._maxiters)

        q_mol.origin_driver_name = 'PYQUANTE'
        cfg = ['atoms={}'.format(atoms),
               'units={}'.format(self._units),
               'charge={}'.format(charge),
               'multiplicity={}'.format(multiplicity),
               'basis={}'.format(basis),
               'hf_method={}'.format(hf_method),
               'tol={}'.format(self._tol),
               'maxiters={}'.format(self._maxiters),
               '']
        q_mol.origin_driver_config = '\n'.join(cfg)

        return q_mol

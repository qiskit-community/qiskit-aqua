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

from typing import Union, List, Optional
import importlib
from enum import Enum
import logging
from qiskit.aqua.utils.validation import validate_min
from ..units_type import UnitsType
from ..fermionic_driver import FermionicDriver, HFMethodType
from ...qiskit_chemistry_error import QiskitChemistryError
from ..molecule import Molecule
from ...qmolecule import QMolecule
from .integrals import compute_integrals

logger = logging.getLogger(__name__)


class BasisType(Enum):
    """ Basis Type """
    BSTO3G = 'sto3g'
    B631G = '6-31g'
    B631GSS = '6-31g**'


class PyQuanteDriver(FermionicDriver):
    """
    Qiskit chemistry driver using the PyQuante2 library.

    See https://github.com/rpmuller/pyquante2
    """

    def __init__(self,
                 atoms: Union[str, List[str]] =
                 'H 0.0 0.0 0.0; H 0.0 0.0 0.735',
                 units: UnitsType = UnitsType.ANGSTROM,
                 charge: int = 0,
                 multiplicity: int = 1,
                 basis: BasisType = BasisType.BSTO3G,
                 hf_method: HFMethodType = HFMethodType.RHF,
                 tol: float = 1e-8,
                 maxiters: int = 100,
                 molecule: Optional[Molecule] = None) -> None:
        """
        Args:
            atoms: Atoms list or string separated by semicolons or line breaks. Each element in the
                list is an atom followed by position e.g. `H 0.0 0.0 0.5`. The preceding example
                shows the `XYZ` format for position but `Z-Matrix` format is supported too here.
            units: Angstrom or Bohr
            charge: Charge on the molecule
            multiplicity: Spin multiplicity (2S+1)
            basis: Basis set; sto3g, 6-31g or 6-31g**
            hf_method: Hartree-Fock Method type
            tol: Convergence tolerance see pyquante2.scf hamiltonians and iterators
            maxiters: Convergence max iterations see pyquante2.scf hamiltonians and iterators,
                has a min. value of 1.
            molecule: A driver independent Molecule definition instance may be provided. When
                a molecule is supplied the `atoms`, `units`, `charge` and `multiplicity` parameters
                are all ignored as the Molecule instance now defines these instead. The Molecule
                object is read when the driver is run and converted to the driver dependent
                configuration for the computation. This allows, for example, the Molecule geometry
                to be updated to compute different points.

        Raises:
            QiskitChemistryError: Invalid Input
        """
        validate_min('maxiters', maxiters, 1)
        self._check_valid()
        if not isinstance(atoms, str) and not isinstance(atoms, list):
            raise QiskitChemistryError("Invalid atom input for PYQUANTE Driver '{}'".format(atoms))

        if isinstance(atoms, list):
            atoms = ';'.join(atoms)
        elif isinstance(atoms, str):
            atoms = atoms.replace('\n', ';')

        super().__init__(molecule=molecule,
                         basis=basis.value,
                         hf_method=hf_method.value,
                         supports_molecule=True)
        self._atoms = atoms
        self._units = units.value
        self._charge = charge
        self._multiplicity = multiplicity
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
        if self.molecule is not None:
            atoms = ';'.join([name + ' ' + ' '.join(map(str, coord))
                              for (name, coord) in self.molecule.geometry])
            charge = self.molecule.charge
            multiplicity = self.molecule.multiplicity
            units = self.molecule.units.value
        else:
            atoms = self._atoms
            charge = self._charge
            multiplicity = self._multiplicity
            units = self._units

        basis = self.basis
        hf_method = self.hf_method

        q_mol = compute_integrals(atoms=atoms,
                                  units=units,
                                  charge=charge,
                                  multiplicity=multiplicity,
                                  basis=basis,
                                  hf_method=hf_method,
                                  tol=self._tol,
                                  maxiters=self._maxiters)

        q_mol.origin_driver_name = 'PYQUANTE'
        cfg = ['atoms={}'.format(atoms),
               'units={}'.format(units),
               'charge={}'.format(charge),
               'multiplicity={}'.format(multiplicity),
               'basis={}'.format(basis),
               'hf_method={}'.format(hf_method),
               'tol={}'.format(self._tol),
               'maxiters={}'.format(self._maxiters),
               '']
        q_mol.origin_driver_config = '\n'.join(cfg)

        return q_mol

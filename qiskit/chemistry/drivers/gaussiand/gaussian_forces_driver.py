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

""" Gaussian Forces Driver """

from typing import Union, List, Optional
import logging

from ..units_type import UnitsType
from ...watson_hamiltonian import WatsonHamiltonian
from ..bosonic_driver import BosonicDriver
from ..molecule import Molecule
from ...qiskit_chemistry_error import QiskitChemistryError
from .gaussian_utils import check_valid
from .gaussian_log_driver import GaussianLogDriver
from .gaussian_log_result import GaussianLogResult

logger = logging.getLogger(__name__)

B3YLP_JCF_DEFAULT = """
#p B3LYP/cc-pVTZ Freq=(Anharm) Int=Ultrafine SCF=VeryTight

CO2 geometry optimization B3LYP/cc-pVTZ

0 1
C        -0.848629    2.067624    0.160992
O         0.098816    2.655801   -0.159738
O        -1.796073    1.479446    0.481721

"""


class GaussianForcesDriver(BosonicDriver):
    """  Gaussian™ 16 forces driver. """

    def __init__(self,
                 jcf: Union[str, List[str]] = B3YLP_JCF_DEFAULT,
                 logfile: Optional[str] = None,
                 molecule: Optional[Molecule] = None,
                 basis: str = 'sto-3g',
                 normalize: bool = True) -> None:
        r"""
        Args:
            jcf: A job control file conforming to Gaussian™ 16 format. This can
                be provided as a single string with '\\n' line separators or as a list of
                strings.
            logfile: Instead of a job control file a log as output from running such a file
                can optionally be given.
            molecule: If a molecule is supplied then an appropriate job control file will be
                built from this, and the `basis`, and will be used in precedence of either the
                `logfile` or the `jcf` params.
            basis: The basis set to be used in the resultant job control file when a
                 molecule is provided.
            normalize: Whether to normalize the factors used in creation of the WatsonHamiltonian
                 as returned when this driver is run.

        Raises:
            QiskitChemistryError: If `jcf` or `molecule` given and Gaussian™ 16 executable
                cannot be located.
        """
        super().__init__(molecule=molecule,
                         basis=basis,
                         hf_method='',
                         supports_molecule=True)
        self._jcf = jcf
        self._logfile = None
        self._normalize = normalize

        # Molecule has precedence if supplied, then an existing logfile
        if self.molecule is None and logfile is not None:
            self._jcf = None
            self._logfile = logfile

        # If running from a jcf or a molecule we need Gaussian™ 16 so check if we have a
        # valid install.
        if self._logfile is None:
            check_valid()

    def run(self) -> WatsonHamiltonian:
        if self._logfile is not None:
            glr = GaussianLogResult(self._logfile)
        else:
            if self._molecule is not None:
                jcf = self._from_molecule_to_str()
            else:
                jcf = self._jcf  # type: ignore
            glr = GaussianLogDriver(jcf=jcf).run()

        return glr.get_watson_hamiltonian(self._normalize)

    def _from_molecule_to_str(self) -> str:
        if self.molecule.units == UnitsType.ANGSTROM:
            units = 'Angstrom'
        elif self.molecule.units == UnitsType.BOHR:
            units = 'Bohr'
        else:
            raise QiskitChemistryError("Unknown unit '{}'".format(self.molecule.units.value))
        cfg1 = f'#p B3LYP/{self.basis} UNITS={units} Freq=(Anharm) Int=Ultrafine SCF=VeryTight\n\n'
        name = ''.join([name for (name, _) in self.molecule.geometry])
        geom = '\n'.join([name + ' ' + ' '.join(map(str, coord))
                          for (name, coord) in self.molecule.geometry])
        cfg2 = f'{name} geometry optimization\n\n'
        cfg3 = f'{self.molecule.charge} {self.molecule.multiplicity}\n{geom}\n\n'
        return cfg1 + cfg2 + cfg3

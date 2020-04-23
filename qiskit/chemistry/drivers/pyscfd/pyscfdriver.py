# -*- coding: utf-8 -*-

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

""" PYSCF Driver """

from typing import Optional, Union, List
import importlib
from enum import Enum
import logging
from qiskit.aqua.utils.validation import validate_min
from qiskit.chemistry.drivers import BaseDriver, UnitsType, HFMethodType
from qiskit.chemistry import QiskitChemistryError, QMolecule
from qiskit.chemistry.drivers.pyscfd.integrals import compute_integrals

logger = logging.getLogger(__name__)


class InitialGuess(Enum):
    """ Initial Guess Enum """
    MINAO = 'minao'
    HCORE = '1e'
    ONE_E = '1e'
    ATOM = 'atom'


class PySCFDriver(BaseDriver):
    """
    Qiskit chemistry driver using the PySCF library.

    See https://sunqm.github.io/pyscf/
    """

    def __init__(self,
                 atom: Union[str, List[str]] = 'H 0.0 0.0 0.0; H 0.0 0.0 0.735',
                 unit: UnitsType = UnitsType.ANGSTROM,
                 charge: int = 0,
                 spin: int = 0,
                 basis: str = 'sto3g',
                 hf_method: HFMethodType = HFMethodType.RHF,
                 conv_tol: float = 1e-9,
                 max_cycle: int = 50,
                 init_guess: InitialGuess = InitialGuess.MINAO,
                 max_memory: Optional[int] = None) -> None:
        """
        Args:
            atom: atom list or string separated by semicolons or line breaks
            unit: angstrom or bohr
            charge: charge
            spin: spin
            basis: basis set
            hf_method: Hartree-Fock Method type
            conv_tol: Convergence tolerance see PySCF docs and pyscf/scf/hf.py
            max_cycle: Max convergence cycles see PySCF docs and pyscf/scf/hf.py,
                       has a min. value of 1.
            init_guess: See PySCF pyscf/scf/hf.py init_guess_by_minao/1e/atom methods
            max_memory: maximum memory

        Raises:
            QiskitChemistryError: Invalid Input
        """
        self._check_valid()
        if not isinstance(atom, list) and not isinstance(atom, str):
            raise QiskitChemistryError("Invalid atom input for PYSCF Driver '{}'".format(atom))

        if isinstance(atom, list):
            atom = ';'.join(atom)
        else:
            atom = atom.replace('\n', ';')

        unit = unit.value
        hf_method = hf_method.value
        init_guess = init_guess.value
        validate_min('max_cycle', max_cycle, 1)
        super().__init__()
        self._atom = atom
        self._unit = unit
        self._charge = charge
        self._spin = spin
        self._basis = basis
        self._hf_method = hf_method
        self._conv_tol = conv_tol
        self._max_cycle = max_cycle
        self._init_guess = init_guess
        self._max_memory = max_memory

    @staticmethod
    def _check_valid():
        err_msg = "PySCF is not installed. See https://sunqm.github.io/pyscf/install.html"
        try:
            spec = importlib.util.find_spec('pyscf')
            if spec is not None:
                return
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug('PySCF check error %s', str(ex))
            raise QiskitChemistryError(err_msg) from ex

        raise QiskitChemistryError(err_msg)

    def run(self) -> QMolecule:
        q_mol = compute_integrals(atom=self._atom,
                                  unit=self._unit,
                                  charge=self._charge,
                                  spin=self._spin,
                                  basis=self._basis,
                                  hf_method=self._hf_method,
                                  conv_tol=self._conv_tol,
                                  max_cycle=self._max_cycle,
                                  init_guess=self._init_guess,
                                  max_memory=self._max_memory)

        q_mol.origin_driver_name = 'PYSCF'
        cfg = ['atom={}'.format(self._atom),
               'unit={}'.format(self._unit),
               'charge={}'.format(self._charge),
               'spin={}'.format(self._spin),
               'basis={}'.format(self._basis),
               'hf_method={}'.format(self._hf_method),
               'conv_tol={}'.format(self._conv_tol),
               'max_cycle={}'.format(self._max_cycle),
               'init_guess={}'.format(self._init_guess),
               'max_memory={}'.format(self._max_memory),
               '']
        q_mol.origin_driver_config = '\n'.join(cfg)

        return q_mol

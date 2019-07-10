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

from qiskit.chemistry.drivers import BaseDriver, UnitsType, HFMethodType
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers.pyscfd.integrals import compute_integrals
import importlib
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InitialGuess(Enum):
    MINAO = 'minao'
    HCORE = '1e'
    ONE_E = '1e'
    ATOM = 'atom'


class PySCFDriver(BaseDriver):
    """Python implementation of a PySCF driver."""

    CONFIGURATION = {
        "name": "PYSCF",
        "description": "PYSCF Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "pyscf_schema",
            "type": "object",
            "properties": {
                "atom": {
                    "type": "string",
                    "default": "H 0.0 0.0 0.0; H 0.0 0.0 0.735"
                },
                "unit": {
                    "type": "string",
                    "default": UnitsType.ANGSTROM.value,
                    "enum": [
                        UnitsType.ANGSTROM.value,
                        UnitsType.BOHR.value,
                    ]
                },
                "charge": {
                    "type": "integer",
                    "default": 0
                },
                "spin": {
                    "type": "integer",
                    "default": 0
                },
                "basis": {
                    "type": "string",
                    "default": "sto3g"
                },
                "hf_method": {
                    "type": "string",
                    "default": HFMethodType.RHF.value,
                    "enum": [
                        HFMethodType.RHF.value,
                        HFMethodType.ROHF.value,
                        HFMethodType.UHF.value
                    ]
                },
                "conv_tol": {
                    "type": "number",
                    "default": 1e-09
                },
                "max_cycle": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1
                },
                "init_guess": {
                    "type": "string",
                    "default": InitialGuess.MINAO.value,
                    "enum": [
                        InitialGuess.MINAO.value,
                        InitialGuess.HCORE.value,
                        InitialGuess.ATOM.value
                    ]
                },
                "max_memory": {
                    "type": ["integer", "null"],
                    "default": None
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self,
                 atom,
                 unit=UnitsType.ANGSTROM,
                 charge=0,
                 spin=0,
                 basis='sto3g',
                 hf_method=HFMethodType.RHF,
                 conv_tol=1e-9,
                 max_cycle=50,
                 init_guess=InitialGuess.MINAO,
                 max_memory=None):
        """
        Initializer
        Args:
            atom (str or list): atom list or string separated by semicolons or line breaks
            unit (UnitsType): angstrom or bohr
            charge (int): charge
            spin (int): spin
            basis (str): basis set
            hf_method (HFMethodType): Hartree-Fock Method type
            conv_tol (float): Convergence tolerance see PySCF docs and pyscf/scf/hf.py
            max_cycle (int): Max convergence cycles see PySCF docs and pyscf/scf/hf.py
            init_guess (InitialGuess): See PySCF pyscf/scf/hf.py init_guess_by_minao/1e/atom methods
            max_memory (int): maximum memory
        """
        if not isinstance(atom, list) and not isinstance(atom, str):
            raise QiskitChemistryError("Invalid atom input for PYSCF Driver '{}'".format(atom))

        if isinstance(atom, list):
            atom = ';'.join(atom)
        else:
            atom = atom.replace('\n', ';')

        unit = unit.value
        hf_method = hf_method.value
        init_guess = init_guess.value
        self.validate(locals())
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
    def check_driver_valid():
        err_msg = "PySCF is not installed. See https://sunqm.github.io/pyscf/install.html"
        try:
            spec = importlib.util.find_spec('pyscf')
            if spec is not None:
                return
        except Exception as e:
            logger.debug('PySCF check error {}'.format(str(e)))
            raise QiskitChemistryError(err_msg) from e

        raise QiskitChemistryError(err_msg)

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        if section is None or not isinstance(section, dict):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        params = section
        kwargs = {}
        for k, v in params.items():
            if k == 'unit':
                v = UnitsType(v)
            elif k == 'hf_method':
                v = HFMethodType(v)
            elif k == 'init_guess':
                v = InitialGuess(v)

            kwargs[k] = v

        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
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

        q_mol.origin_driver_name = self.configuration['name']
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

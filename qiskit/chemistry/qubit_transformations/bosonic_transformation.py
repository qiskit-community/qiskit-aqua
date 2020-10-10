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
"""
This module implements a vibronic Hamiltonian operator, representing the
energy of the nuclei in a molecule.
"""

import logging
from typing import Tuple, List, Union, Any, Optional
from enum import Enum

from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import WatsonHamiltonian
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.bosonic_operator import BosonicOperator
from qiskit.chemistry.results import EigenstateResult, VibronicStructureResult
from qiskit.chemistry.components.bosonic_basis import HarmonicBasis
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from .qubit_operator_transformation import QubitOperatorTransformation

logger = logging.getLogger(__name__)


class BosonicTransformationType(Enum):
    """ BosonicTransformationType enum """
    HARMONIC = 'harmonic'


class BosonicQubitMappingType(Enum):
    """ BosonicQubitMappingType enum """
    DIRECT = 'direct'


class BosonicTransformation(QubitOperatorTransformation):
    """A vibronic Hamiltonian operator representing the energy of the nuclei in the molecule"""

    def __init__(self, qubit_mapping: BosonicQubitMappingType = BosonicQubitMappingType.DIRECT,
                 transformation_type: BosonicTransformationType = BosonicTransformationType.HARMONIC,
                 basis_size: Union[int, List[int]] = 2,
                 truncation: int = 3):
        """
        Args:
            qubit_mapping: a string giving the type of mapping (only the 'direct' mapping is
                implemented at this point)
            transformation_type: a string giving the modal basis.
                The Hamiltonian is expressed in this basis.
            basis_size: define the number of modals per mode. If the number of modals is the
                same for each mode, then only an int is required.
                However, if the number of modals differ depending on the mode basis_size should be
                a list of int, for example: [3,4] means 2 modes: first mode has 3 modals,
                second mode has 4 modals.
            truncation: where is the Hamiltonian expansion truncation (1 for having only
                              1-body terms, 2 for having on 1- and 2-body terms...)
        """

        self._qubit_mapping = qubit_mapping.value
        self._transformation_type = transformation_type.value
        self._basis_size = basis_size
        self._truncation_order = truncation

        self._num_modes = None
        self._h_mat = None

    @property
    def num_modes(self):
        """
        Returns: the number of modes
        """
        return self._num_modes

    def transform(self, driver: BaseDriver,
                  aux_operators: Optional[List[Any]] = None
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """
        Transformation to qubit operator from the driver

        Args:
            driver: BaseDriver

        Returns:
            qubit operator, auxiliary operators
        """
        watson = driver.run()
        ops, aux_ops = self._do_transform(watson, aux_operators)

        return ops, aux_ops


    def _do_transform(self, watson: WatsonHamiltonian,
                      aux_operators: Optional[List[Union[BosonicOperator,
                                                         WeightedPauliOperator]]] = None
                      ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:

        self._num_modes = watson.num_modes

        if self._transformation_type == 'harmonic':
            if isinstance(self._basis_size, int):
                self._basis_size = [self._basis_size] * self._num_modes
            self._h_mat = HarmonicBasis(watson, self._basis_size, self._truncation_order).run()
        else:
            raise QiskitChemistryError('Unknown Transformation type')

        bos_op = BosonicOperator(self._h_mat, self._basis_size)
        qubit_op = bos_op.mapping(qubit_mapping = self._qubit_mapping)
        qubit_op.name = 'Bosonic Operator'

        aux_ops = []

        def _add_aux_op(aux_op: BosonicOperator, name: str) -> None:
            """
            Add auxiliary operators

            Args:
                aux_op: auxiliary operators
                name: name

            """
            if not isinstance(aux_op, WeightedPauliOperator):
                aux_qop = BosonicTransformation._map_fermionic_operator_to_qubit(
                    aux_op, self._qubit_mapping)
                aux_qop.name = name
            else:
                aux_qop = aux_op

            aux_ops.append(aux_qop)
            logger.debug('  num paulis: %s', aux_qop.paulis)

        logger.debug('Creating aux op for number of occupied modals per mode')

        for mode in range(self._num_modes):
            _add_aux_op(bos_op.number_occupied_modals_per_mode(mode),
                        'Number of occupied modals in mode {}'.format(mode))

        # add user specified auxiliary operators
        if aux_operators is not None:
            for aux_op in aux_operators:
                _add_aux_op(aux_op, aux_op.name)

        return qubit_op, aux_ops

    def interpret(self, eigenstate_result: EigenstateResult) -> VibronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            eigenstate_result: an eigenstate result object.

        Returns:
            A vibronic structure result.
        """
        result = VibronicStructureResult(eigenstate_result.data)
        result.computed_vibronic_energy = eigenstate_result.eigenvalue.real
        if result.aux_operator_eigenvalues is not None:
            occ_modals = []
            for mode in range(self._num_modes):
                if result.aux_operator_eigenvalues[mode] is not None:
                    occ_modals.append(result.aux_operator_eigenvalues[mode][0].real)
                else:
                    occ_modals.append(None)
            result.num_occupied_modals_per_mode = occ_modals

        return result

    @staticmethod
    def _map_fermionic_operator_to_qubit(bos_op: BosonicOperator,
                                         qubit_mapping: str) -> WeightedPauliOperator:
        """

        Args:
            bos_op: a BosonicOperator
            qubit_mapping: the type of boson to qubit mapping

        Returns: qubit operator

        """

        qubit_op = bos_op.mapping(qubit_mapping=qubit_mapping, threshold=0.00001)

        return qubit_op



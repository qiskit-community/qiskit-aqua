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
"""
This module implements a vibronic Hamiltonian operator, representing the
energy of the nuclei in a molecule.
"""

import logging
from enum import Enum
from functools import partial
from typing import Tuple, List, Union, Any, Optional, Callable, Dict

import numpy as np

from qiskit.tools import parallel_map
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry import WatsonHamiltonian
from qiskit.chemistry.bosonic_operator import BosonicOperator
from qiskit.chemistry.components.bosonic_bases import HarmonicBasis
from qiskit.chemistry.components.variational_forms import UVCC
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import EigenstateResult, VibronicStructureResult
from .transformation import Transformation

logger = logging.getLogger(__name__)


class BosonicTransformationType(Enum):
    """ BosonicTransformationType enum """
    HARMONIC = 'harmonic'


class BosonicQubitMappingType(Enum):
    """ BosonicQubitMappingType enum """
    DIRECT = 'direct'


class BosonicTransformation(Transformation):
    """A vibronic Hamiltonian operator representing the energy of the nuclei in the molecule"""

    def __init__(self, qubit_mapping: BosonicQubitMappingType = BosonicQubitMappingType.DIRECT,
                 transformation_type:
                 BosonicTransformationType = BosonicTransformationType.HARMONIC,
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
        self._untapered_qubit_op = None

    @property
    def num_modes(self) -> int:
        """
        Returns: the number of modes
        """
        return self._num_modes

    @property
    def basis(self) -> Union[int, List[int]]:
        """ returns the basis (number of modals per mode) """
        return self._basis_size

    @property
    def commutation_rule(self) -> bool:
        """Getter of the commutation rule"""
        return True

    @property
    def untapered_qubit_op(self):
        """Getter for the untapered qubit operator"""
        return self._untapered_qubit_op

    def transform(self, driver: BaseDriver,
                  aux_operators: Optional[List[Any]] = None
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """
        Transformation to qubit operator from the driver

        Args:
            driver: BaseDriver
            aux_operators: Optional additional aux ops to evaluate
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
            self._h_mat = HarmonicBasis(watson, self._basis_size, self._truncation_order).convert()
        else:
            raise QiskitChemistryError('Unknown Transformation type')

        bos_op = BosonicOperator(self._h_mat, self._basis_size)
        qubit_op = bos_op.mapping(qubit_mapping=self._qubit_mapping)
        self._untapered_qubit_op = qubit_op
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
                aux_qop = BosonicTransformation._map_bosonic_operator_to_qubit(
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

    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> VibronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

               Args:
                   raw_result: an eigenstate result object.

               Returns:
                   An vibronic structure result.
               """

        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues

        result = VibronicStructureResult(eigenstate_result.data)
        result.computed_vibronic_energies = eigenstate_result.eigenenergies
        if result.aux_operator_eigenvalues is not None:
            if not isinstance(result.aux_operator_eigenvalues, list):
                aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
            else:
                aux_operator_eigenvalues = result.aux_operator_eigenvalues

            result.num_occupied_modals_per_mode = []
            for aux_op_eigenvalues in aux_operator_eigenvalues:
                occ_modals = []
                for mode in range(self._num_modes):
                    if aux_op_eigenvalues[mode] is not None:
                        occ_modals.append(aux_op_eigenvalues[mode][0].real)
                    else:
                        occ_modals.append(None)
                result.num_occupied_modals_per_mode.append(occ_modals)

        return result

    @staticmethod
    def _map_bosonic_operator_to_qubit(bos_op: BosonicOperator,
                                       qubit_mapping: str) -> WeightedPauliOperator:
        """
        Args:
            bos_op: a BosonicOperator
            qubit_mapping: the type of boson to qubit mapping

        Returns:
            qubit operator
        """

        qubit_op = bos_op.mapping(qubit_mapping=qubit_mapping, threshold=0.00001)

        return qubit_op

    @staticmethod
    def _build_single_hopping_operator(index: List[List[int]],
                                       basis: List[int],
                                       qubit_mapping: str) -> WeightedPauliOperator:
        """
        Builds a hopping operator given the list of indices (index) that is a single, a double
        or a higher order excitation.

        Args:
            index: the indexes defining the excitation
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode basis = [4,4,4]
            qubit_mapping: the qubits mapping type. Only 'direct' is supported at the moment.

        Returns:
            Qubit operator object corresponding to the hopping operator
        """
        degree = len(index)
        hml = []  # type: List[List]
        for _ in range(degree):
            hml.append([])

        tmp = []
        for i in range(len(index))[::-1]:
            tmp.append(index[i])

        hml[-1].append([tmp, 1])

        dummpy_op = BosonicOperator(np.asarray(hml, dtype=object), basis)
        qubit_op = dummpy_op.mapping(qubit_mapping)
        if len(qubit_op.paulis) == 0:
            qubit_op = None

        return qubit_op

    def build_hopping_operators(self, excitations: Union[str, List[List[int]]] = 'sd') \
            -> Tuple[Dict[str, WeightedPauliOperator],
                     Dict,
                     Dict[str, List[List[int]]]]:
        """
        Args:
            excitations:

        Returns:
            Dict of hopping operators, dict of commutativity types and dict of excitation indices
        """
        exctn_types = {'s': 0, 'd': 1}

        if isinstance(excitations, str):
            degrees = [exctn_types[letter] for letter in excitations]
            excitations_list = UVCC.compute_excitation_lists(self._basis_size, degrees)
        else:
            excitations_list = excitations

        size = len(excitations_list)

        def _dag_list(extn_lst):
            dag_lst = []
            for lst in extn_lst:
                dag_lst.append([lst[0], lst[2], lst[1]])
            return dag_lst

        hopping_operators: Dict[str, WeightedPauliOperator] = {}
        excitation_indices = {}
        to_be_executed_list = []
        for idx in range(size):
            to_be_executed_list += [excitations_list[idx], _dag_list(excitations_list[idx])]
            hopping_operators['E_{}'.format(idx)] = None
            hopping_operators['Edag_{}'.format(idx)] = None
            excitation_indices['E_{}'.format(idx)] = excitations_list[idx]
            excitation_indices['Edag_{}'.format(idx)] = _dag_list(excitations_list[idx])

        result = parallel_map(self._build_single_hopping_operator,
                              to_be_executed_list,
                              task_args=(self._basis_size,
                                         self._qubit_mapping),
                              num_processes=aqua_globals.num_processes)

        for key, res in zip(hopping_operators.keys(), result):
            hopping_operators[key] = res

        # This variable is required for compatibility with the FermionicTransformation
        # at the moment we do not have any type of commutativity in the bosonic case.
        type_of_commutativities: Dict[str, List[bool]] = {}

        return hopping_operators, type_of_commutativities, excitation_indices

    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            # the first num_modes aux_value is the evaluated number of particles for the given mode
            for mode in range(self._num_modes):
                if not np.isclose(aux_values[mode][0], 1):
                    return False
            return True

        return partial(filter_criterion, self)

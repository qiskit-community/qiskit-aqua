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

"""This module implements a transformation from a fermionic problem to a qubit operator.

The problem is described in a driver.
"""

from functools import partial
from typing import Optional, List, Union, cast, Tuple, Dict, Any, Callable
import logging
from enum import Enum

import numpy as np
from qiskit.tools import parallel_map
from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator, OperatorBase
from qiskit.aqua.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.chemistry import QiskitChemistryError, QMolecule
from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import DipoleTuple, EigenstateResult, ElectronicStructureResult
from qiskit.chemistry.components.variational_forms import UCCSD

from .transformation import Transformation
from ..components.initial_states import HartreeFock

logger = logging.getLogger(__name__)


class FermionicTransformationType(Enum):
    """ Electronic Transformation Type enum """
    FULL = 'full'
    PARTICLE_HOLE = 'particle_hole'


class FermionicQubitMappingType(Enum):
    """ FermionicQubitMappingType enum """
    JORDAN_WIGNER = 'jordan_wigner'
    PARITY = 'parity'
    BRAVYI_KITAEV = 'bravyi_kitaev'


class FermionicTransformation(Transformation):
    """A transformation from a fermionic problem, represented by a driver, to a qubit operator."""

    def __init__(self,
                 transformation: FermionicTransformationType = FermionicTransformationType.FULL,
                 qubit_mapping: FermionicQubitMappingType = FermionicQubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:
        """
        Args:
            transformation: full or particle_hole
            qubit_mapping: 'jordan_wigner', 'parity' or 'bravyi_kitaev'
            two_qubit_reduction: Whether two qubit reduction should be used,
                                        when parity mapping only
            freeze_core: Whether to freeze core orbitals when possible
            orbital_reduction: Orbital list to be frozen or removed
            z2symmetry_reduction: If z2 symmetry reduction should be applied to resulting
                qubit operators that are computed. For each symmetry detected the operator will be
                split in two where each requires one qubit less for computation. So for example
                3 symmetries will split in the original operator into 8 new operators each
                requiring 3 less qubits. Now only one of these operators will have the ground state
                and be the correct symmetry sector needed for the ground state. Setting 'auto' will
                use an automatic computation of the correct sector. If from other experiments, with
                the z2symmetry logic, the sector is known, then the tapering values of that sector
                can be provided (a list of int of values -1, and 1). The default is None
                meaning no symmetry reduction is done. Note that dipole and other operators
                such as spin, num particles etc are also symmetry reduced according to the
                symmetries found in the main operator if this operator commutes with the main
                operator symmetry. If it does not then the operator will be discarded since no
                meaningful measurement can take place.
        Raises:
            QiskitChemistryError: Invalid symmetry reduction
        """
        transformation = transformation.value
        orbital_reduction = orbital_reduction if orbital_reduction is not None else []
        super().__init__()
        self._transformation = transformation
        self._qubit_mapping = qubit_mapping.value
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction
        if z2symmetry_reduction is not None:
            if isinstance(z2symmetry_reduction, str):
                if z2symmetry_reduction != 'auto':
                    raise QiskitChemistryError('Invalid z2symmetry_reduction value')
        self._z2symmetry_reduction = z2symmetry_reduction
        self._has_dipole_moments = False
        self._untapered_qubit_op = None

        # Store values that are computed by the classical logic in order
        # that later they may be combined with the quantum result
        self._hf_energy = None
        self._nuclear_repulsion_energy = None
        self._nuclear_dipole_moment = None
        self._reverse_dipole_sign = None
        # The following shifts are from freezing orbitals under orbital reduction
        self._energy_shift = 0.0
        self._x_dipole_shift = 0.0
        self._y_dipole_shift = 0.0
        self._z_dipole_shift = 0.0
        # The following shifts are from particle_hole transformation
        self._ph_energy_shift = 0.0
        self._ph_x_dipole_shift = 0.0
        self._ph_y_dipole_shift = 0.0
        self._ph_z_dipole_shift = 0.0

        self._molecule_info: Dict[str, Any] = {}

    @property
    def commutation_rule(self) -> bool:
        """Getter of the commutation rule"""
        return False

    @property
    def molecule_info(self) -> Dict[str, Any]:
        """Getter of the molecule information."""
        return self._molecule_info

    @property
    def qubit_mapping(self) -> str:
        """Getter of the qubit mapping."""
        return self._qubit_mapping

    def transform(self, driver: BaseDriver,
                  aux_operators: Optional[List[FermionicOperator]] = None
                  ) -> Tuple[OperatorBase, List[OperatorBase]]:
        """Transformation from the ``driver`` to a qubit operator.

        Args:
            driver: A driver encoding the molecule information.
            aux_operators: Additional auxiliary ``FermionicOperator``s to evaluate.

        Returns:
            A qubit operator and a dictionary of auxiliary operators.
        """
        q_molecule = driver.run()
        ops, aux_ops = self._do_transform(q_molecule, aux_operators)

        # the internal method may still return legacy operators which is why we make sure to convert
        # all of the operator to the operator flow
        ops = ops.to_opflow() if isinstance(ops, WeightedPauliOperator) else ops
        aux_ops = [a.to_opflow() if isinstance(a, WeightedPauliOperator) else a for a in aux_ops]

        return ops, aux_ops

    def _do_transform(self, qmolecule: QMolecule,
                      aux_operators: Optional[List[FermionicOperator]] = None
                      ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """
        Args:
            qmolecule: qmolecule
            aux_operators: Additional ``FermionicOperator``s to map to a qubit operator.

        Returns:
            (qubit operator, auxiliary operators)

        """
        logger.debug('Processing started...')
        # Save these values for later combination with the quantum computation result
        self._hf_energy = qmolecule.hf_energy
        self._nuclear_repulsion_energy = qmolecule.nuclear_repulsion_energy
        self._nuclear_dipole_moment = qmolecule.nuclear_dipole_moment
        self._reverse_dipole_sign = qmolecule.reverse_dipole_sign

        core_list = qmolecule.core_orbitals if self._freeze_core else []
        reduce_list = self._orbital_reduction

        if self._freeze_core:
            logger.info("Freeze_core specified. Core orbitals to be frozen: %s", core_list)
        if reduce_list:
            logger.info("Configured orbital reduction list: %s", reduce_list)
            reduce_list = [x + qmolecule.num_orbitals if x < 0 else x for x in reduce_list]

        freeze_list = []
        remove_list = []

        # Orbitals are specified by their index from 0 to n-1, where n is the number of orbitals the
        # molecule has. The combined list of the core orbitals, when freeze_core is true, with any
        # user supplied orbitals is what will be used. Negative numbers may be used to indicate the
        # upper virtual orbitals, so -1 is the highest, then -2 etc. and these will
        # be converted to the
        # positive 0-based index for computation.
        # In the combined list any orbitals that are occupied are added to a freeze list and an
        # energy is stored from these orbitals to be added later.
        # Unoccupied orbitals are just discarded.
        # Because freeze and eliminate is done in separate steps,
        # with freeze first, we have to re-base
        # the indexes for elimination according to how many orbitals were removed when freezing.
        #
        orbitals_list = list(set(core_list + reduce_list))
        num_alpha = qmolecule.num_alpha
        num_beta = qmolecule.num_beta
        new_num_alpha = num_alpha
        new_num_beta = num_beta
        if orbitals_list:
            orbitals_list = np.array(orbitals_list)
            orbitals_list = orbitals_list[(cast(np.ndarray, orbitals_list) >= 0) &
                                          (orbitals_list < qmolecule.num_orbitals)]

            freeze_list_alpha = [i for i in orbitals_list if i < num_alpha]
            freeze_list_beta = [i for i in orbitals_list if i < num_beta]
            freeze_list = np.append(freeze_list_alpha,
                                    [i + qmolecule.num_orbitals for i in freeze_list_beta])

            remove_list_alpha = [i for i in orbitals_list if i >= num_alpha]
            remove_list_beta = [i for i in orbitals_list if i >= num_beta]
            rla_adjust = -len(freeze_list_alpha)
            rlb_adjust = -len(freeze_list_alpha) - len(freeze_list_beta) + qmolecule.num_orbitals
            remove_list = np.append([i + rla_adjust for i in remove_list_alpha],
                                    [i + rlb_adjust for i in remove_list_beta])

            logger.info("Combined orbital reduction list: %s", orbitals_list)
            logger.info("  converting to spin orbital reduction list: %s",
                        np.append(np.array(orbitals_list),
                                  np.array(orbitals_list) + qmolecule.num_orbitals))
            logger.info("    => freezing spin orbitals: %s", freeze_list)
            logger.info("    => removing spin orbitals: %s (indexes accounting for freeze %s)",
                        np.append(remove_list_alpha,
                                  np.array(remove_list_beta) + qmolecule.num_orbitals), remove_list)

            new_num_alpha -= len(freeze_list_alpha)
            new_num_beta -= len(freeze_list_beta)

        new_nel = [new_num_alpha, new_num_beta]

        # construct the fermionic operator
        fer_op = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)

        # try to reduce it according to the freeze and remove list
        fer_op, self._energy_shift, did_shift = \
            FermionicTransformation._try_reduce_fermionic_operator(fer_op, freeze_list, remove_list)
        # apply same transformation for the aux operators
        if aux_operators is not None:
            aux_operators = [
                FermionicTransformation._try_reduce_fermionic_operator(
                    op, freeze_list, remove_list)[0]
                for op in aux_operators
                ]

        if did_shift:
            logger.info("Frozen orbital energy shift: %s", self._energy_shift)
        # apply particle hole transformation, if specified
        if self._transformation == FermionicTransformationType.PARTICLE_HOLE.value:
            fer_op, ph_shift = fer_op.particle_hole_transformation(new_nel)
            self._ph_energy_shift = -ph_shift
            logger.info("Particle hole energy shift: %s", self._ph_energy_shift)

            # apply the same transformation for the aux operators
            if aux_operators is not None:
                aux_operators = [
                    op.particle_hole_transformation(new_nel)[0]
                    for op in aux_operators
                    ]

        logger.debug('Converting to qubit using %s mapping', self._qubit_mapping)
        qubit_op = FermionicTransformation._map_fermionic_operator_to_qubit(
            fer_op, self._qubit_mapping, new_nel, self._two_qubit_reduction
            )
        qubit_op.name = 'Fermionic Operator'

        logger.debug('  num paulis: %s, num qubits: %s', len(qubit_op.paulis), qubit_op.num_qubits)

        aux_ops = []  # list of the aux operators

        def _add_aux_op(aux_op: FermionicOperator, name: str) -> None:
            """
            Add auxiliary operators

            Args:
                aux_op: auxiliary operators
                name: name

            """
            aux_qop = FermionicTransformation._map_fermionic_operator_to_qubit(
                aux_op, self._qubit_mapping, new_nel, self._two_qubit_reduction
                )
            aux_qop.name = name

            aux_ops.append(aux_qop)
            logger.debug('  num paulis: %s', aux_qop.paulis)

        # the first three operators are hardcoded to number of particles, angular momentum
        # and magnetization in this order
        logger.debug('Creating aux op for Number of Particles')
        _add_aux_op(fer_op.total_particle_number(), 'Number of Particles')
        logger.debug('Creating aux op for S^2')
        _add_aux_op(fer_op.total_angular_momentum(), 'S^2')
        logger.debug('Creating aux op for Magnetization')
        _add_aux_op(fer_op.total_magnetization(), 'Magnetization')

        # the next three are dipole moments, if supported by the qmolecule
        if qmolecule.has_dipole_integrals():
            self._has_dipole_moments = True

            def _dipole_op(dipole_integrals: np.ndarray, axis: str) \
                    -> Tuple[WeightedPauliOperator, float, float]:
                """
                Dipole operators

                Args:
                    dipole_integrals: dipole integrals
                    axis: axis for dipole moment calculation

                Returns:
                    (qubit_op_, shift, ph_shift_)
                """
                logger.debug('Creating aux op for dipole %s', axis)
                fer_op_ = FermionicOperator(h1=dipole_integrals)
                fer_op_, shift, did_shift_ = self._try_reduce_fermionic_operator(fer_op_,
                                                                                 freeze_list,
                                                                                 remove_list)
                if did_shift_:
                    logger.info("Frozen orbital %s dipole shift: %s", axis, shift)
                ph_shift_ = 0.0
                if self._transformation == FermionicTransformationType.PARTICLE_HOLE.value:
                    fer_op_, ph_shift_ = fer_op_.particle_hole_transformation(new_nel)
                    ph_shift_ = -ph_shift_
                    logger.info("Particle hole %s dipole shift: %s", axis, ph_shift_)
                qubit_op_ = self._map_fermionic_operator_to_qubit(fer_op_,
                                                                  self._qubit_mapping,
                                                                  new_nel,
                                                                  self._two_qubit_reduction)
                qubit_op_.name = 'Dipole ' + axis
                logger.debug('  num paulis: %s', len(qubit_op_.paulis))
                return qubit_op_, shift, ph_shift_

            op_dipole_x, self._x_dipole_shift, self._ph_x_dipole_shift = \
                _dipole_op(qmolecule.x_dipole_integrals, 'x')
            op_dipole_y, self._y_dipole_shift, self._ph_y_dipole_shift = \
                _dipole_op(qmolecule.y_dipole_integrals, 'y')
            op_dipole_z, self._z_dipole_shift, self._ph_z_dipole_shift = \
                _dipole_op(qmolecule.z_dipole_integrals, 'z')

            aux_ops += [op_dipole_x, op_dipole_y, op_dipole_z]

        # add user specified auxiliary operators
        if aux_operators is not None:
            for aux_op in aux_operators:
                if hasattr(aux_op, 'name'):
                    name = aux_op.name
                else:
                    name = ''
                _add_aux_op(aux_op, name)

        logger.info('Molecule num electrons: %s, remaining for processing: %s',
                    [num_alpha, num_beta], new_nel)
        nspinorbs = qmolecule.num_orbitals * 2
        new_nspinorbs = nspinorbs - len(freeze_list) - len(remove_list)
        logger.info('Molecule num spin orbitals: %s, remaining for processing: %s',
                    nspinorbs, new_nspinorbs)

        self._molecule_info['num_particles'] = [new_num_alpha, new_num_beta]
        self._molecule_info['num_orbitals'] = new_nspinorbs
        reduction = self._two_qubit_reduction if self._qubit_mapping == 'parity' else False
        self._molecule_info['two_qubit_reduction'] = reduction
        self._untapered_qubit_op = qubit_op

        z2symmetries = Z2Symmetries([], [], [], None)
        if self._z2symmetry_reduction is not None:
            logger.debug('Processing z2 symmetries')
            qubit_op, aux_ops, z2symmetries = self._process_z2symmetry_reduction(qubit_op, aux_ops)
        self._molecule_info['z2_symmetries'] = z2symmetries

        logger.debug('Processing complete ready to run algorithm')
        return qubit_op, aux_ops

    @property
    def untapered_qubit_op(self):
        """Getter for the untapered qubit operator"""
        return self._untapered_qubit_op

    def _process_z2symmetry_reduction(self,
                                      qubit_op: WeightedPauliOperator,
                                      aux_ops: List[WeightedPauliOperator]) -> Tuple:
        """
        Implement z2 symmetries in the qubit operator

        Args:
            qubit_op : qubit operator
            aux_ops: auxiliary operators

        Returns:
            (z2_qubit_op, z2_aux_ops, z2_symmetries)

        Raises:
            QiskitChemistryError: Invalid input
        """
        z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
        if z2_symmetries.is_empty():
            logger.debug('No Z2 symmetries found')
            z2_qubit_op = qubit_op
            z2_aux_ops = aux_ops
            z2_symmetries = Z2Symmetries([], [], [], None)
        else:
            logger.debug('%s Z2 symmetries found: %s', len(z2_symmetries.symmetries),
                         ','.join([symm.to_label() for symm in z2_symmetries.symmetries]))

            # Check auxiliary operators commute with main operator's symmetry
            logger.debug('Checking operators commute with symmetry:')
            symmetry_ops = []
            for symmetry in z2_symmetries.symmetries:
                symmetry_ops.append(WeightedPauliOperator(paulis=[[1.0, symmetry]]))
            commutes = FermionicTransformation._check_commutes(symmetry_ops, qubit_op)
            if not commutes:
                raise QiskitChemistryError('Z2 symmetry failure main operator must commute '
                                           'with symmetries found from it')
            for i, aux_op in enumerate(aux_ops):
                commutes = FermionicTransformation._check_commutes(symmetry_ops, aux_op)
                if not commutes:
                    aux_ops[i] = None  # Discard since no meaningful measurement can be done

            if self._z2symmetry_reduction == 'auto':
                hf_state = HartreeFock(num_orbitals=self._molecule_info['num_orbitals'],
                                       qubit_mapping=self._qubit_mapping,
                                       two_qubit_reduction=self._two_qubit_reduction,
                                       num_particles=self._molecule_info['num_particles'])
                z2_symmetries = FermionicTransformation._pick_sector(z2_symmetries, hf_state.bitstr)
            else:
                if len(self._z2symmetry_reduction) != len(z2_symmetries.symmetries):
                    raise QiskitChemistryError('z2symmetry_reduction tapering values list has '
                                               'invalid length {} should be {}'.
                                               format(len(self._z2symmetry_reduction),
                                                      len(z2_symmetries.symmetries)))
                valid = np.all(np.isin(self._z2symmetry_reduction, [-1, 1]))
                if not valid:
                    raise QiskitChemistryError('z2symmetry_reduction tapering values list must '
                                               'contain -1\'s and/or 1\'s only was {}'.
                                               format(self._z2symmetry_reduction,))
                z2_symmetries.tapering_values = self._z2symmetry_reduction

            logger.debug('Apply symmetry with tapering values %s', z2_symmetries.tapering_values)
            chop_to = 0.00000001  # Use same threshold as qubit mapping to chop tapered operator
            z2_qubit_op = z2_symmetries.taper(qubit_op).chop(chop_to)
            z2_aux_ops = []
            for aux_op in aux_ops:
                if aux_op is None:
                    z2_aux_ops += [None]
                else:
                    z2_aux_ops += [z2_symmetries.taper(aux_op).chop(chop_to)]

        return z2_qubit_op, z2_aux_ops, z2_symmetries

    @staticmethod
    def _check_commutes(cliffords: List[WeightedPauliOperator],
                        operator: WeightedPauliOperator) -> bool:
        """
        Check commutations

        Args:
            cliffords : cliffords
            operator: qubit operator

        Returns:
            Boolean: does_commute
        """
        commutes = []
        for clifford in cliffords:
            commutes.append(operator.commute_with(clifford))
        does_commute = np.all(commutes)
        logger.debug('  \'%s\' commutes: %s, %s', operator.name, does_commute, commutes)
        return does_commute

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
            # the first aux_value is the evaluated number of particles
            num_particles_aux = aux_values[0][0]
            # the second aux_value is the total angular momentum which (for singlets) should be zero
            total_angular_momentum_aux = aux_values[1][0]
            return np.isclose(sum(self.molecule_info['num_particles']), num_particles_aux) and \
                np.isclose(0., total_angular_momentum_aux)

        return partial(filter_criterion, self)

    @staticmethod
    def _pick_sector(z2_symmetries: Z2Symmetries, hf_str: np.ndarray) -> Z2Symmetries:
        """
        Based on Hartree-Fock bit string and found symmetries to determine the sector.
        The input z2 symmetries will be mutated with the determined tapering values.

        Args:
            z2_symmetries: the z2 symmetries object.
            hf_str: Hartree-Fock bit string (the last index is for qubit 0).

        Returns:
            the original z2 symmetries filled with the correct tapering values.
        """
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coef = []
        for sym in z2_symmetries.symmetries:
            # pylint: disable=no-member
            coef = -1 if np.logical_xor.reduce(np.logical_and(sym.z[::-1], hf_str)) else 1
            taper_coef.append(coef)
        z2_symmetries.tapering_values = taper_coef
        return z2_symmetries

    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
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
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]

        result = ElectronicStructureResult(eigenstate_result.data)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        result.hartree_fock_energy = self._hf_energy
        result.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        if self._nuclear_dipole_moment is not None:
            result.nuclear_dipole_moment = tuple(x for x in self._nuclear_dipole_moment)
        result.ph_extracted_energy = self._ph_energy_shift
        result.frozen_extracted_energy = self._energy_shift
        if result.aux_operator_eigenvalues is not None:
            # the first three values are hardcoded to number of particles, angular momentum
            # and magnetization in this order
            result.num_particles = []
            result.total_angular_momentum = []
            result.magnetization = []
            result.computed_dipole_moment = []
            result.ph_extracted_dipole_moment = []
            result.frozen_extracted_dipole_moment = []
            if not isinstance(result.aux_operator_eigenvalues, list):
                aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
            else:
                aux_operator_eigenvalues = result.aux_operator_eigenvalues
            for aux_op_eigenvalues in aux_operator_eigenvalues:
                if aux_op_eigenvalues is None:
                    continue
                if aux_op_eigenvalues[0] is not None:
                    result.num_particles.append(aux_op_eigenvalues[0][0].real)

                if aux_op_eigenvalues[1] is not None:
                    result.total_angular_momentum.append(aux_op_eigenvalues[1][0].real)

                if aux_op_eigenvalues[2] is not None:
                    result.magnetization.append(aux_op_eigenvalues[2][0].real)

                # the next three are hardcoded to Dipole moments, if they are set
                if len(aux_op_eigenvalues) >= 6 and self._has_dipole_moments:
                    # check if the names match
                    # extract dipole moment in each axis
                    dipole_moment = []
                    for moment in aux_op_eigenvalues[3:6]:
                        if moment is not None:
                            dipole_moment += [moment[0].real]
                        else:
                            dipole_moment += [None]

                    result.reverse_dipole_sign = self._reverse_dipole_sign
                    result.computed_dipole_moment.append(cast(DipoleTuple,
                                                              tuple(dipole_moment)))
                    result.ph_extracted_dipole_moment.append(
                        (self._ph_x_dipole_shift, self._ph_y_dipole_shift,
                         self._ph_z_dipole_shift))
                    result.frozen_extracted_dipole_moment.append(
                        (self._x_dipole_shift, self._y_dipole_shift,
                         self._z_dipole_shift))

        return result

    @staticmethod
    def _try_reduce_fermionic_operator(fer_op: FermionicOperator,
                                       freeze_list: List,
                                       remove_list: List) -> Tuple:
        """
        Trying to reduce the fermionic operator w.r.t to freeze and remove list if provided

        Args:
            fer_op: fermionic operator
            freeze_list: freeze list of orbitals
            remove_list: remove list of orbitals

        Returns:
            (fermionic_operator, energy_shift, did_shift)
        """
        # pylint: disable=len-as-condition
        did_shift = False
        energy_shift = 0.0
        if len(freeze_list) > 0:
            fer_op, energy_shift = fer_op.fermion_mode_freezing(freeze_list)
            did_shift = True
        if len(remove_list) > 0:
            fer_op = fer_op.fermion_mode_elimination(remove_list)
        return fer_op, energy_shift, did_shift

    @staticmethod
    def _map_fermionic_operator_to_qubit(fer_op: FermionicOperator,
                                         qubit_mapping: str,
                                         num_particles: List[int],
                                         two_qubit_reduction: bool) -> WeightedPauliOperator:
        """

        Args:
            fer_op: Fermionic Operator
            qubit_mapping: fermionic to qubit mapping
            num_particles: number of particles
            two_qubit_reduction: two qubit reduction

        Returns:
            qubit operator
        """

        qubit_op = fer_op.mapping(map_type=qubit_mapping, threshold=0.00000001)
        if qubit_mapping == 'parity' and two_qubit_reduction:
            qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)
        return qubit_op

    @staticmethod
    def _build_single_hopping_operator(index, num_particles, num_orbitals, qubit_mapping,
                                       two_qubit_reduction, z2_symmetries):

        h_1 = np.zeros((num_orbitals, num_orbitals), dtype=complex)
        h_2 = np.zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals), dtype=complex)
        if len(index) == 2:
            i, j = index
            h_1[i, j] = 4.0
        elif len(index) == 4:
            i, j, k, m = index
            h_2[i, j, k, m] = 16.0
        fer_op = FermionicOperator(h_1, h_2)
        qubit_op = fer_op.mapping(qubit_mapping)
        if qubit_mapping == 'parity' and two_qubit_reduction:
            qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)

        commutativities = []
        if not z2_symmetries.is_empty():
            for symmetry in z2_symmetries.symmetries:
                symmetry_op = WeightedPauliOperator(paulis=[[1.0, symmetry]])
                commuting = qubit_op.commute_with(symmetry_op)
                anticommuting = qubit_op.anticommute_with(symmetry_op)

                if commuting != anticommuting:  # only one of them is True
                    if commuting:
                        commutativities.append(True)
                    elif anticommuting:
                        commutativities.append(False)
                else:
                    raise AquaError("Symmetry {} is nor commute neither anti-commute "
                                    "to exciting operator.".format(symmetry.to_label()))

        return qubit_op, commutativities

    def build_hopping_operators(self, excitations: Union[str, List[List[int]]] = 'sd'
                                ) -> Tuple[Dict[str, WeightedPauliOperator],
                                           Dict[str, List[bool]],
                                           Dict[str, List[Any]]]:
        """Builds the product of raising and lowering operators (basic excitation operators)

        Args:
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise a list of custom excitations can directly be provided.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """

        num_alpha, num_beta = self._molecule_info['num_particles']
        num_orbitals = self._molecule_info['num_orbitals']

        if isinstance(excitations, str):
            se_list, de_list = UCCSD.compute_excitation_lists([num_alpha, num_beta], num_orbitals,
                                                              excitation_type=excitations)
            excitations_list = se_list+de_list
        else:
            excitations_list = excitations

        size = len(excitations_list)

        # # get all to-be-processed index
        # mus, nus = np.triu_indices(size)

        # build all hopping operators
        hopping_operators: Dict[str, WeightedPauliOperator] = {}
        type_of_commutativities: Dict[str, List[bool]] = {}
        excitation_indices = {}
        to_be_executed_list = []
        for idx in range(size):
            to_be_executed_list += [excitations_list[idx], list(reversed(excitations_list[idx]))]
            hopping_operators['E_{}'.format(idx)] = None
            hopping_operators['Edag_{}'.format(idx)] = None
            type_of_commutativities['E_{}'.format(idx)] = None
            type_of_commutativities['Edag_{}'.format(idx)] = None
            excitation_indices['E_{}'.format(idx)] = excitations_list[idx]
            excitation_indices['Edag_{}'.format(idx)] = list(reversed(excitations_list[idx]))

        result = parallel_map(self._build_single_hopping_operator,
                              to_be_executed_list,
                              task_args=(num_alpha + num_beta,
                                         num_orbitals,
                                         self._qubit_mapping,
                                         self._two_qubit_reduction,
                                         self._molecule_info['z2_symmetries']),
                              num_processes=aqua_globals.num_processes)

        for key, res in zip(hopping_operators.keys(), result):
            hopping_operators[key] = res[0]
            type_of_commutativities[key] = res[1]

        return hopping_operators, type_of_commutativities, excitation_indices

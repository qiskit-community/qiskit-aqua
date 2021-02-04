# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module implements a molecular Hamiltonian operator, representing the
energy of the electrons and nuclei in a molecule.
"""
import warnings
from typing import Optional, List, Union, cast, Tuple
import logging
from enum import Enum

import numpy as np
from qiskit.aqua.algorithms import MinimumEigensolverResult, EigensolverResult
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator
from qiskit.chemistry import QMolecule, QiskitChemistryError
from qiskit.chemistry.fermionic_operator import FermionicOperator
from .chemistry_operator import (ChemistryOperator,
                                 MolecularGroundStateResult,
                                 DipoleTuple)

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """ Transformation Type enum """
    FULL = 'full'
    PARTICLE_HOLE = 'particle_hole'


class QubitMappingType(Enum):
    """ QubitMappingType enum """
    JORDAN_WIGNER = 'jordan_wigner'
    PARITY = 'parity'
    BRAVYI_KITAEV = 'bravyi_kitaev'


class Hamiltonian(ChemistryOperator):
    """
    A molecular Hamiltonian operator, representing the
    energy of the electrons and nuclei in a molecule.
    """

    def __init__(self,
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:
        """
        Args:
            transformation: full or particle_hole
            qubit_mapping: jordan_wigner, parity or bravyi_kitaev
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
        warnings.warn('The Hamiltonian class is deprecated as of Qiskit Aqua 0.8.0 and will be '
                      'removed no earlier than 3 months after the release date. Instead, the '
                      'FermionicTransformation can be used.', DeprecationWarning, stacklevel=2)
        orbital_reduction = orbital_reduction if orbital_reduction is not None else []
        super().__init__()
        self._transformation = transformation.value
        self._qubit_mapping = qubit_mapping.value
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction
        if z2symmetry_reduction is not None:
            if isinstance(z2symmetry_reduction, str):
                if z2symmetry_reduction != 'auto':
                    raise QiskitChemistryError('Invalid z2symmetry_reduction value')
        self._z2symmetry_reduction = z2symmetry_reduction

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

    def run(self, qmolecule: QMolecule) -> Tuple[WeightedPauliOperator,
                                                 List[WeightedPauliOperator]]:
        """ run method"""
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
        orb_list = list(set(core_list + reduce_list))
        num_alpha = qmolecule.num_alpha
        num_beta = qmolecule.num_beta
        new_num_alpha = num_alpha
        new_num_beta = num_beta
        if orb_list:
            orbitals_list = np.array(orb_list)
            orbitals_list = \
                orbitals_list[(cast(np.ndarray, orbitals_list) >= 0) &
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

        fer_op = FermionicOperator(h1=qmolecule.one_body_integrals, h2=qmolecule.two_body_integrals)
        fer_op, self._energy_shift, did_shift = \
            Hamiltonian._try_reduce_fermionic_operator(fer_op, freeze_list, remove_list)
        if did_shift:
            logger.info("Frozen orbital energy shift: %s", self._energy_shift)
        if self._transformation == TransformationType.PARTICLE_HOLE.value:
            fer_op, ph_shift = fer_op.particle_hole_transformation(new_nel)
            self._ph_energy_shift = -ph_shift
            logger.info("Particle hole energy shift: %s", self._ph_energy_shift)
        logger.debug('Converting to qubit using %s mapping', self._qubit_mapping)
        qubit_op = Hamiltonian._map_fermionic_operator_to_qubit(fer_op,
                                                                self._qubit_mapping, new_nel,
                                                                self._two_qubit_reduction)
        qubit_op.name = 'Electronic Hamiltonian'

        logger.debug('  num paulis: %s, num qubits: %s', len(qubit_op.paulis), qubit_op.num_qubits)

        aux_ops = []

        def _add_aux_op(aux_op, name):
            aux_qop = Hamiltonian._map_fermionic_operator_to_qubit(aux_op,
                                                                   self._qubit_mapping,
                                                                   new_nel,
                                                                   self._two_qubit_reduction)
            aux_qop.name = name
            aux_ops.append(aux_qop)
            logger.debug('  num paulis: %s', aux_qop.paulis)

        logger.debug('Creating aux op for Number of Particles')
        _add_aux_op(fer_op.total_particle_number(), 'Number of Particles')
        logger.debug('Creating aux op for S^2')
        _add_aux_op(fer_op.total_angular_momentum(), 'S^2')
        logger.debug('Creating aux op for Magnetization')
        _add_aux_op(fer_op.total_magnetization(), 'Magnetization')

        if qmolecule.has_dipole_integrals():
            def _dipole_op(dipole_integrals, axis):
                logger.debug('Creating aux op for dipole %s', axis)
                fer_op_ = FermionicOperator(h1=dipole_integrals)
                fer_op_, shift, did_shift_ = self._try_reduce_fermionic_operator(fer_op_,
                                                                                 freeze_list,
                                                                                 remove_list)
                if did_shift_:
                    logger.info("Frozen orbital %s dipole shift: %s", axis, shift)
                ph_shift_ = 0.0
                if self._transformation == TransformationType.PARTICLE_HOLE.value:
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

            aux_ops.append(op_dipole_x)
            aux_ops.append(op_dipole_y)
            aux_ops.append(op_dipole_z)

        logger.info('Molecule num electrons: %s, remaining for processing: %s',
                    [num_alpha, num_beta], new_nel)
        nspinorbs = qmolecule.num_orbitals * 2
        new_nspinorbs = nspinorbs - len(freeze_list) - len(remove_list)
        logger.info('Molecule num spin orbitals: %s, remaining for processing: %s',
                    nspinorbs, new_nspinorbs)

        self._add_molecule_info(self.INFO_NUM_PARTICLES, (new_num_alpha, new_num_beta))
        self._add_molecule_info(self.INFO_NUM_ORBITALS, new_nspinorbs)
        self._add_molecule_info(self.INFO_TWO_QUBIT_REDUCTION,
                                self._two_qubit_reduction
                                if self._qubit_mapping == 'parity' else False)

        z2symmetries = Z2Symmetries([], [], [], None)
        if self._z2symmetry_reduction is not None:
            logger.debug('Processing z2 symmetries')
            qubit_op, aux_ops, z2symmetries = self._process_z2symmetry_reduction(qubit_op, aux_ops)
        self._add_molecule_info(self.INFO_Z2SYMMETRIES, z2symmetries)

        logger.debug('Processing complete ready to run algorithm')
        return qubit_op, aux_ops

    def _process_z2symmetry_reduction(self, qubit_op, aux_ops):

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
            commutes = Hamiltonian._check_commutes(symmetry_ops, qubit_op)
            if not commutes:
                raise QiskitChemistryError('Z2 symmetry failure main operator must commute '
                                           'with symmetries found from it')
            for i, aux_op in enumerate(aux_ops):
                commutes = Hamiltonian._check_commutes(symmetry_ops, aux_op)
                if not commutes:
                    aux_ops[i] = None  # Discard since no meaningful measurement can be done

            if self._z2symmetry_reduction == 'auto':
                from ..circuit.library.initial_states.hartree_fock import hartree_fock_bitstring
                hf_bitstr = hartree_fock_bitstring(
                    num_orbitals=self._molecule_info['num_orbitals'],
                    qubit_mapping=self._qubit_mapping,
                    two_qubit_reduction=self._two_qubit_reduction,
                    num_particles=self._molecule_info['num_particles']
                    )
                z2_symmetries = Hamiltonian._pick_sector(z2_symmetries, hf_bitstr)
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
                z2_aux_ops.append(z2_symmetries.taper(aux_op).chop(chop_to) if aux_op is not None
                                  else None)

        return z2_qubit_op, z2_aux_ops, z2_symmetries

    @staticmethod
    def _check_commutes(cliffords, operator):
        commutes = []
        for clifford in cliffords:
            commutes.append(operator.commute_with(clifford))
        does_commute = np.all(commutes)
        logger.debug('  \'%s\' commutes: %s, %s', operator.name, does_commute, commutes)
        return does_commute

    @staticmethod
    def _pick_sector(z2_symmetries, hf_str):
        """
        Based on Hartree-Fock bit string and found symmetries to determine the sector.
        The input z2 symmetries will be mutated with the determined tapering values.

        Args:
            z2_symmetries (Z2Symmetries): the z2 symmetries object.
            hf_str (numpy.ndarray): Hartree-Fock bit string (the last index is for qubit 0).

        Returns:
            Z2Symmetries: the original z2 symmetries filled with the correct tapering values.
        """
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coef = []
        for sym in z2_symmetries.symmetries:
            # pylint: disable=no-member
            coef = -1 if np.logical_xor.reduce(np.logical_and(sym.z[::-1], hf_str)) else 1
            taper_coef.append(coef)
        z2_symmetries.tapering_values = taper_coef
        return z2_symmetries

    # Called by public superclass method process_algorithm_result to complete specific processing
    def _process_algorithm_result(self, algo_result):
        if isinstance(algo_result, MinimumEigensolverResult):
            return self._process_algorithm_result_ground_state(algo_result)
        elif isinstance(algo_result, EigensolverResult):
            return self._process_algorithm_result_deprecated(algo_result)
            # TODO return self._process_algorithm_result_excited_states(algo_result)
        else:
            return self._process_algorithm_result_deprecated(algo_result)

    def _process_algorithm_result_ground_state(self, algo_result: MinimumEigensolverResult) \
            -> MolecularGroundStateResult:
        mgsr = MolecularGroundStateResult()
        mgsr.algorithm_result = algo_result
        mgsr.hartree_fock_energy = self._hf_energy
        mgsr.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        if self._nuclear_dipole_moment is not None:
            mgsr.nuclear_dipole_moment = tuple(x for x in self._nuclear_dipole_moment)
        mgsr.computed_electronic_energy = algo_result.eigenvalue.real
        mgsr.ph_extracted_energy = self._ph_energy_shift
        mgsr.frozen_extracted_energy = self._energy_shift
        aux_ops_vals = algo_result.aux_operator_eigenvalues
        if aux_ops_vals is not None:
            # Dipole results if dipole aux ops were present
            dipole_idx = 3
            if len(aux_ops_vals) > dipole_idx:
                mgsr.reverse_dipole_sign = self._reverse_dipole_sign
                dipm = []
                for i in range(dipole_idx, dipole_idx+3):  # Gets X, Y and Z components
                    dipm.append(aux_ops_vals[i][0].real if aux_ops_vals[i] is not None else None)
                mgsr.computed_dipole_moment = cast(DipoleTuple, tuple(dipm))
                mgsr.ph_extracted_dipole_moment = (self._ph_x_dipole_shift,
                                                   self._ph_y_dipole_shift,
                                                   self._ph_z_dipole_shift)
                mgsr.frozen_extracted_dipole_moment = (self._x_dipole_shift,
                                                       self._y_dipole_shift,
                                                       self._z_dipole_shift)
            # The first 3 entries are num particles, total angular momentum and magnetization
            mgsr.num_particles = aux_ops_vals[0][0].real \
                if aux_ops_vals[0] is not None else None
            mgsr.total_angular_momentum = aux_ops_vals[1][0].real \
                if aux_ops_vals[1] is not None else None
            mgsr.magnetization = aux_ops_vals[2][0].real \
                if aux_ops_vals[2] is not None else None
        return mgsr

    def _process_algorithm_result_deprecated(self, algo_result):
        warnings.warn('Processing a dictionary result is deprecated,'
                      ' pass a (minimum) eigensolver result now.', DeprecationWarning)
        # pylint: disable=len-as-condition
        result = {}

        # Ground state energy
        egse = algo_result['energy'] + self._energy_shift + self._ph_energy_shift
        result['energy'] = egse
        lines = ['=== GROUND STATE ENERGY ===']
        lines.append(' ')
        lines.append('* Electronic ground state energy (Hartree): {}'.format(round(egse, 12)))
        lines.append('  - computed part:      {}'.format(round(algo_result['energy'], 12)))
        lines.append('  - frozen energy part: {}'.format(round(self._energy_shift, 12)))
        lines.append('  - particle hole part: {}'.format(round(self._ph_energy_shift, 12)))
        if self._nuclear_repulsion_energy is not None:
            lines.append('~ Nuclear repulsion energy (Hartree): {}'.format(
                round(self._nuclear_repulsion_energy, 12)))
            lines.append('> Total ground state energy (Hartree): {}'.format(
                round(self._nuclear_repulsion_energy + egse, 12)))
            if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0:
                aux_ops = algo_result['aux_ops'][0]
                num_particles = aux_ops[0][0]
                spin_squared = aux_ops[1][0]
                spin = (-1.0 + np.sqrt(1 + 4 * spin_squared)) / 2
                m = aux_ops[2][0]
                lines.append(
                    '  Measured:: Num particles: {:.3f}, S: {:.3f}, M: {:.5f}'.format(
                        num_particles, spin, m))
            result['energy'] = self._nuclear_repulsion_energy + egse
            result['nuclear_repulsion_energy'] = self._nuclear_repulsion_energy
        if self._hf_energy is not None:
            result['hf_energy'] = self._hf_energy

        # Excited states list - it includes ground state too
        if 'energies' in algo_result:
            exsce = \
                [x + self._energy_shift + self._ph_energy_shift for x in algo_result['energies']]
            exste = [x + self._nuclear_repulsion_energy for x in exsce]
            result['energies'] = exste
            if len(exsce) > 1:
                lines.append(' ')
                lines.append('=== EXCITED STATES ===')
                lines.append(' ')
                lines.append(
                    '> Excited states energies (plus ground): {}'.format(
                        [round(x, 12) for x in exste]))
                lines.append(
                    '    - computed: {}'.format([round(x, 12) for x in algo_result['energies']]))
                if 'cond_number' in algo_result:  # VQKE condition num for eigen vals
                    lines.append('    - cond num: {}'.format(algo_result['cond_number']))

                if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0:
                    lines.append(
                        '  ......................................................................')
                    lines.append(
                        '  ###:  Total Energy,      Computed,       # particles,   S         M')
                    for i in range(len(algo_result['aux_ops'])):
                        aux_ops = algo_result['aux_ops'][i]
                        num_particles = aux_ops[0][0]
                        spin_squared = aux_ops[1][0]
                        spin = (-1.0 + np.sqrt(1 + 4 * spin_squared)) / 2
                        m = aux_ops[2][0]
                        lines.append(
                            '  {:>3}: {: 16.12f}, {: 16.12f},     {:5.3f},   {:5.3f},  {:8.5f}'.
                            format(i, exste[i], algo_result['energies'][i], num_particles, spin, m))
        else:
            result['energies'] = [result['energy']]

        # Dipole computation
        dipole_idx = 3
        if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0 and \
                len(algo_result['aux_ops'][0]) > dipole_idx:
            dipole_moments_x = algo_result['aux_ops'][0][dipole_idx + 0][0]
            dipole_moments_y = algo_result['aux_ops'][0][dipole_idx + 1][0]
            dipole_moments_z = algo_result['aux_ops'][0][dipole_idx + 2][0]

            _elec_dipole = \
                np.array([dipole_moments_x + self._x_dipole_shift + self._ph_x_dipole_shift,
                          dipole_moments_y + self._y_dipole_shift + self._ph_y_dipole_shift,
                          dipole_moments_z + self._z_dipole_shift + self._ph_z_dipole_shift])
            lines.append(' ')
            lines.append('=== DIPOLE MOMENT ===')
            lines.append(' ')
            lines.append('* Electronic dipole moment (a.u.): {}'.format(
                Hamiltonian._dipole_to_string(_elec_dipole)))
            lines.append('  - computed part:      {}'.format(
                Hamiltonian._dipole_to_string([dipole_moments_x,
                                               dipole_moments_y, dipole_moments_z])))
            lines.append('  - frozen energy part: {}'.format(
                Hamiltonian._dipole_to_string([self._x_dipole_shift,
                                               self._y_dipole_shift, self._z_dipole_shift])))
            lines.append('  - particle hole part: {}'.format(
                Hamiltonian._dipole_to_string([self._ph_x_dipole_shift,
                                               self._ph_y_dipole_shift, self._ph_z_dipole_shift])))
            if self._nuclear_dipole_moment is not None:
                if self._reverse_dipole_sign:
                    _elec_dipole = -_elec_dipole
                dipole_moment = self._nuclear_dipole_moment + _elec_dipole
                total_dipole_moment = np.sqrt(np.sum(np.power(dipole_moment, 2)))
                lines.append('~ Nuclear dipole moment (a.u.): {}'.format(
                    Hamiltonian._dipole_to_string(self._nuclear_dipole_moment)))
                lines.append('> Dipole moment (a.u.): {}  Total: {}'.format(
                    Hamiltonian._dipole_to_string(dipole_moment),
                    Hamiltonian._float_to_string(total_dipole_moment)))
                lines.append('               (debye): {}  Total: {}'.format(
                    Hamiltonian._dipole_to_string(dipole_moment / QMolecule.DEBYE),
                    Hamiltonian._float_to_string(total_dipole_moment / QMolecule.DEBYE)))
                result['nuclear_dipole_moment'] = self._nuclear_dipole_moment
                result['electronic_dipole_moment'] = _elec_dipole
                result['dipole_moment'] = dipole_moment
                result['total_dipole_moment'] = total_dipole_moment

        return lines, result

    @staticmethod
    def _try_reduce_fermionic_operator(fer_op, freeze_list, remove_list):
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
    def _map_fermionic_operator_to_qubit(fer_op, qubit_mapping, num_particles, two_qubit_reduction):
        qubit_op = fer_op.mapping(map_type=qubit_mapping, threshold=0.00000001)
        if qubit_mapping == 'parity' and two_qubit_reduction:
            qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)
        return qubit_op

    @staticmethod
    def _dipole_to_string(_dipole):
        dips = [round(x, 8) for x in _dipole]
        value = '['
        for i, _ in enumerate(dips):
            value += Hamiltonian._float_to_string(dips[i])
            value += '  ' if i < len(dips) - 1 else ']'
        return value

    @staticmethod
    def _float_to_string(value, precision=8):
        return '0.0' if value == 0 else ('{:.' + str(precision) + 'f}').format(value).rstrip('0')

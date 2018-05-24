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
# =============================================================================
"""
This module implements a molecular Hamiltonian operator, representing the
energy of the electrons and nuclei in a molecule.
"""

from .chemistry_operator import ChemistryOperator
from qiskit_acqua_chemistry.fermionic_operator import FermionicOperator
from qiskit_acqua.input.energyinput import EnergyInput
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Hamiltonian(ChemistryOperator):
    """
    A molecular Hamiltonian operator, representing the
    energy of the electrons and nuclei in a molecule.
    """

    KEY_TRANSFORMATION = 'transformation'
    KEY_QUBIT_MAPPING = 'qubit_mapping'
    KEY_TWO_QUBIT_REDUCTION = 'two_qubit_reduction'
    KEY_FREEZE_CORE = 'freeze_core'
    KEY_ORBITAL_REDUCTION = 'orbital_reduction'
    KEY_MAX_WORKERS = 'max_workers'

    TRANSFORMATION_FULL = 'full'
    TRANSFORMATION_PH = 'particle_hole'
    QUBIT_MAPPING_JORDAN_WIGNER = 'jordan_wigner'
    QUBIT_MAPPING_PARITY = 'parity'
    QUBIT_MAPPING_BINARY_TREE = 'binary_tree'

    HAMILTONIAN_CONFIGURATION = {
        'name': 'hamiltonian',
        'description': 'Hamiltonian chemistry operator',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hamiltonian_schema',
            'type': 'object',
            'properties': {
                'transformation': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'particle_hole']}
                    ]
                },
                'qubit_mapping': {
                    'type': 'string',
                    'default': 'parity',
                    'oneOf': [
                        {'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']}
                    ]
                },
                'two_qubit_reduction': {
                    'type': 'boolean',
                    'default': True
                },
                'freeze_core': {
                    'type': 'boolean',
                    'default': False
                },
                'orbital_reduction': {
                    'default': [],
                    'type': 'array',
                    'items': {
                        'type': 'number'
                    }
                },
                'max_workers': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                }
            },
            "additionalProperties": False
        },
        'problems': ['energy', 'excited_states']
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.HAMILTONIAN_CONFIGURATION.copy())
        self._transformation = 'full'
        self._qubit_mapping = 'parity'
        self._two_qubit_reduction = True
        self._freeze_core = False
        self._orbital_reduction = []
        self._max_workers = 4

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

    def init_args(self, transformation='full', qubit_mapping='parity', two_qubit_reduction=True,
                  freeze_core=False, orbital_reduction=[], max_workers=999):
        """
        Initial according to schema params
        Args:
            transformation (str): full or particle_hole
            qubit_mapping (str: jordan_wigner, parity or bravyi_kitaev
            two_qubit_reduction (bool): Whether two qubit reduction should be used, when parity mapping only
            freeze_core: Whether to freeze core orbitals when possible
            orbital_reduction: Orbital list to be frozen or removed
            max_workers: Max workers processes for transformation

        Returns:

        """
        self._transformation = transformation
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction
        self._max_workers = max_workers

    def run(self, qmolecule):
        logger.debug('Processing started...')
        # Save these values for later combination with the quantum computation result
        self._hf_energy = qmolecule._hf_energy
        self._nuclear_repulsion_energy = qmolecule._nuclear_repulsion_energy
        self._nuclear_dipole_moment = qmolecule._nuclear_dipole_moment
        self._reverse_dipole_sign = qmolecule._reverse_dipole_sign

        core_list = qmolecule.core_orbitals if self._freeze_core else []
        reduce_list = self._orbital_reduction

        if self._freeze_core:
            logger.info("Freeze_core specified. Core orbitals to be frozen: {}".format(core_list))
        if len(reduce_list) > 0:
            logger.info("Configured orbital reduction list: {}".format(reduce_list))
            reduce_list = [x + qmolecule._num_orbitals if x < 0 else x for x in reduce_list]

        freeze_list = []
        remove_list = []

        # Orbitals are specified by their index from 0 to n-1, where n is the number of orbitals the
        # molecule has. The combined list of the core orbitals, when freeze_core is true, with any
        # user supplied orbitals is what will be used. Negative numbers may be used to indicate the
        # upper virtual orbitals, so -1 is the highest, then -2 etc. and these will be converted to the
        # positive 0-based index for computation.
        # In the combined list any orbitals that are occupied are added to a freeze list and an
        # energy is stored from these orbitals to be added later. Unoccupied orbitals are just discarded.
        # Because freeze and eliminate is done in separate steps, with freeze first, we have to re-base
        # the indexes for elimination according to how many orbitals were removed when freezing.
        #
        orbitals_list = list(set(core_list + reduce_list))
        nel = qmolecule._num_alpha + qmolecule._num_beta
        new_nel = nel
        if len(orbitals_list) > 0:
            orbitals_list = np.array(orbitals_list)
            orbitals_list = orbitals_list[(orbitals_list >= 0) & (orbitals_list < qmolecule._num_orbitals)]

            freeze_list = [i for i in orbitals_list if i < int(nel/2)]
            freeze_list = np.append(np.array(freeze_list), np.array(freeze_list) + qmolecule._num_orbitals)

            remove_list = [i for i in orbitals_list if i >= int(nel/2)]
            remove_list_orig_idx = np.append(np.array(remove_list), np.array(remove_list) + qmolecule._num_orbitals)
            remove_list = np.append(np.array(remove_list) - int(len(freeze_list)/2), np.array(remove_list) + qmolecule._num_orbitals - len(freeze_list))
            logger.info("Combined orbital reduction list: {}".format(orbitals_list))
            logger.info("  converting to spin orbital reduction list: {}".format(np.append(np.array(orbitals_list), np.array(orbitals_list) + qmolecule._num_orbitals)))
            logger.info("    => freezing spin orbitals: {}".format(freeze_list))
            logger.info("    => removing spin orbitals: {} (indexes accounting for freeze {})".format(remove_list_orig_idx, remove_list))

            new_nel -= len(freeze_list)

        fer_op = FermionicOperator(h1=qmolecule._one_body_integrals, h2=qmolecule._two_body_integrals)
        fer_op, self._energy_shift, did_shift = Hamiltonian._try_reduce_fermionic_operator(fer_op, freeze_list, remove_list)
        if did_shift:
            logger.info("Frozen orbital energy shift: {}".format(self._energy_shift))
        if self._transformation == Hamiltonian.TRANSFORMATION_PH:
            fer_op, ph_shift = fer_op.particle_hole_transformation(new_nel)
            self._ph_energy_shift = -ph_shift
            logger.info("Particle hole energy shift: {}".format(self._ph_energy_shift))
        logger.debug('Converting to qubit using {} mapping'.format(self._qubit_mapping))
        qubit_op = Hamiltonian._map_fermionic_operator_to_qubit(fer_op, self._qubit_mapping, new_nel,
                                                                self._two_qubit_reduction, self._max_workers)
        algo_input = EnergyInput()
        algo_input.qubit_op = qubit_op

        def _add_aux_op(aux_op):
            algo_input.add_aux_op(Hamiltonian._map_fermionic_operator_to_qubit(aux_op, self._qubit_mapping, new_nel,
                                                                               self._two_qubit_reduction, self._max_workers))
        logger.debug('Creating aux op for Number of Particles')
        _add_aux_op(fer_op.total_particle_number())
        logger.debug('Creating aux op for S^2')
        _add_aux_op(fer_op.total_angular_momentum())
        logger.debug('Creating aux op for Magnetization')
        _add_aux_op(fer_op.total_magnetization())

        if qmolecule.has_dipole_integrals():
            def _dipole_op(dipole_integrals, axis):
                logger.debug('Creating aux op for dipole {}'.format(axis))
                fer_op_ = FermionicOperator(h1=dipole_integrals)
                fer_op_, shift, did_shift_ = self._try_reduce_fermionic_operator(fer_op_, freeze_list, remove_list)
                if did_shift_:
                    logger.info("Frozen orbital {} dipole shift: {}".format(axis, shift))
                ph_shift_ = 0.0
                if self._transformation == Hamiltonian.TRANSFORMATION_PH:
                    fer_op_, ph_shift_ = fer_op_.particle_hole_transformation(new_nel)
                    ph_shift_ = -ph_shift_
                    logger.info("Particle hole {} dipole shift: {}".format(axis, ph_shift_))
                qubit_op_ = self._map_fermionic_operator_to_qubit(fer_op_, self._qubit_mapping, new_nel,
                                                                  self._two_qubit_reduction, self._max_workers)
                return qubit_op_, shift, ph_shift_

            op_dipole_x, self._x_dipole_shift, self._ph_x_dipole_shift = _dipole_op(qmolecule._x_dipole_integrals, 'x')
            op_dipole_y, self._y_dipole_shift, self._ph_y_dipole_shift = _dipole_op(qmolecule._y_dipole_integrals, 'y')
            op_dipole_z, self._z_dipole_shift, self._ph_z_dipole_shift = _dipole_op(qmolecule._z_dipole_integrals, 'z')

            algo_input.add_aux_op(op_dipole_x)
            algo_input.add_aux_op(op_dipole_y)
            algo_input.add_aux_op(op_dipole_z)

        logger.info('Molecule num electrons: {}, remaining for processing: {}'.format(nel, new_nel))
        nspinorbs = qmolecule._num_orbitals * 2
        new_nspinorbs = nspinorbs - len(freeze_list) - len(remove_list)
        logger.info('Molecule num spin orbitals: {}, remaining for processing: {}'.format(nspinorbs, new_nspinorbs))

        self._add_molecule_info(self.INFO_NUM_PARTICLES, new_nel)
        self._add_molecule_info(self.INFO_NUM_ORBITALS, new_nspinorbs)
        self._add_molecule_info(self.INFO_TWO_QUBIT_REDUCTION,
                                self._two_qubit_reduction if self._qubit_mapping == 'parity' else False)

        logger.debug('Processing complete ready to run algorithm')
        return algo_input

    # Called by public superclass method process_algorithm_result to complete specific processing
    def _process_algorithm_result(self, algo_result):
        result = {}

        # Ground state energy
        egse = algo_result['energy'] + self._energy_shift + self._ph_energy_shift
        result['energy'] = egse
        lines = ['* Electronic ground state energy: {}'.format(egse)]
        lines.append('  - computed part:      {}'.format(algo_result['energy']))
        lines.append('  - frozen energy part: {}'.format(self._energy_shift))
        lines.append('  - particle hole part: {}'.format(self._ph_energy_shift))
        if self._nuclear_repulsion_energy is not None:
            lines.append('~ Nuclear repulsion energy: {}'.format(self._nuclear_repulsion_energy))
            lines.append('> Total ground state energy: {}'.format(self._nuclear_repulsion_energy + egse))
            if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0:
                aux_ops = algo_result['aux_ops'][0]
                num_particles = aux_ops[0][0]
                s_squared = aux_ops[1][0]
                s = (-1.0 + np.sqrt(1 + 4 * s_squared)) / 2
                m = aux_ops[2][0]
                lines.append('  Measured:: Num particles: {:.3f}, S: {:.3f}, M: {:.5f}'.format(num_particles, s, m))
            result['energy'] = self._nuclear_repulsion_energy + egse
            result['nuclear_repulsion_energy'] = self._nuclear_repulsion_energy
        if self._hf_energy is not None:
            result['hf_energy'] = self._hf_energy

        # Excited states list - it includes ground state too
        if 'energies' in algo_result:
            exsce = [x + self._energy_shift + self._ph_energy_shift for x in algo_result['energies']]
            exste = [x + self._nuclear_repulsion_energy for x in exsce]
            result['energies'] = exste
            if len(exsce) > 1:
                lines.append('> Excited states energies (plus ground): {}'.format(exste))
                lines.append('    - computed: {}'.format(algo_result['energies']))
                if 'cond_number' in algo_result:  # VQKE condition num for eigen vals
                    lines.append('    - cond num: {}'.format(algo_result['cond_number']))

                if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0:
                    lines.append(' ')
                    lines.append('  ###:  Energy,            Computed,       # particles,   S         M')
                    for i in range(len(algo_result['aux_ops'])):
                        aux_ops = algo_result['aux_ops'][i]
                        num_particles = aux_ops[0][0]
                        s_squared = aux_ops[1][0]
                        s = (-1.0 + np.sqrt(1 + 4 * s_squared)) / 2
                        m = aux_ops[2][0]
                        lines.append('  {:>3}: {: 16.12f}, {: 16.12f},     {:5.3f},   {:5.3f},  {:8.5f}'
                                     .format(i, exste[i], algo_result['energies'][i], num_particles, s, m))
        else:
            result['energies'] = [result['energy']]

        # Dipole computation
        dipole_idx = 3
        if 'aux_ops' in algo_result and len(algo_result['aux_ops']) > 0 and len(algo_result['aux_ops'][0]) > dipole_idx:
            dipole_moments_x = algo_result['aux_ops'][0][dipole_idx+0][0]
            dipole_moments_y = algo_result['aux_ops'][0][dipole_idx+1][0]
            dipole_moments_z = algo_result['aux_ops'][0][dipole_idx+2][0]

            _elec_dipole = np.array([dipole_moments_x + self._x_dipole_shift + self._ph_x_dipole_shift,
                                     dipole_moments_y + self._y_dipole_shift + self._ph_y_dipole_shift,
                                     dipole_moments_z + self._z_dipole_shift + self._ph_z_dipole_shift])
            lines.append('* Electronic dipole moment: {}'.format(_elec_dipole))
            lines.append('  - computed part:      {}'.format(np.array([dipole_moments_x, dipole_moments_y, dipole_moments_z])))
            lines.append('  - frozen energy part: {}'.format(np.array([self._x_dipole_shift, self._y_dipole_shift, self._z_dipole_shift])))
            lines.append('  - particle hole part: {}'.format(np.array([self._ph_x_dipole_shift, self._ph_y_dipole_shift, self._ph_z_dipole_shift])))
            if self._nuclear_dipole_moment is not None:
                if self._reverse_dipole_sign:
                    _elec_dipole = -_elec_dipole
                dipole_moment = self._nuclear_dipole_moment + _elec_dipole
                total_dipole_moment = np.sqrt(np.sum(np.power(dipole_moment, 2)))
                lines.append('~ Nuclear dipole moment: {}'.format(self._nuclear_dipole_moment))
                lines.append('> Dipole moment: {}  Total: {}'.format(dipole_moment, total_dipole_moment))
                result['nuclear_dipole_moment'] = self._nuclear_dipole_moment
                result['electronic_dipole_moment'] = _elec_dipole
                result['dipole_moment'] = dipole_moment
                result['total_dipole_moment'] = total_dipole_moment

        return lines, result

    @staticmethod
    def _try_reduce_fermionic_operator(fer_op, freeze_list, remove_list):
        did_shift = False
        energy_shift = 0.0
        if len(freeze_list) > 0:
            fer_op, energy_shift = fer_op.fermion_mode_freezing(freeze_list)
            did_shift = True
        if len(remove_list) > 0:
            fer_op = fer_op.fermion_mode_elimination(remove_list)
        return fer_op, energy_shift, did_shift

    @staticmethod
    def _map_fermionic_operator_to_qubit(fer_op, qubit_mapping, num_particles, two_qubit_reduction, max_workers):
        qubit_op = fer_op.mapping(map_type=qubit_mapping, threshold=0.00000001, num_workers=max_workers)
        if qubit_mapping == 'parity' and two_qubit_reduction:
            qubit_op = qubit_op.two_qubit_reduced_operator(num_particles)
        return qubit_op

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" TODO """

import numpy as np
import logging
import itertools
import sys

from typing import Optional, List

from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import Z2Symmetries, commutator
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.excited_states_calculation import qEOMExcitedStatesCalculation

logger = logging.getLogger(__name__)


class NumericalqEOMExcitedStatesCalculation(qEOMExcitedStatesCalculation):

    def __init__(self, ground_state_calculation: GroundStateCalculation,
                 excitation_type: str = 'SD', active_space: Optional[List[int]] = None):
        super().__init__(ground_state_calculation,excitation_type,active_space)

    def prepare_matrix_operators(self):

        hopping_operators, type_of_commutativities = self._gsc.transformation.build_hopping_operators(
            self._excitation_type, self._active_space) # already need the number of qubits etc

        eom_matrix_operators = self.build_all_commutators(hopping_operators, type_of_commutativities)

        return eom_matrix_operators

    def build_all_commutators(self, hopping_operators, type_of_commutativities):
        """Building all commutators for Q, W, M, V matrices.

        Args:
            excitations_list (list): single excitations list + double excitation list
            hopping_operators (dict): all hopping operators based on excitations_list,
                                      key is the string of single/double excitation;
                                      value is corresponding operator.
            type_of_commutativities (dict): if tapering is used, it records the commutativities of
                                     hopping operators with the
                                     Z2 symmetries found in the original operator.
        Returns:
            dict: key: a string of matrix indices; value: the commutators for Q matrix
            dict: key: a string of matrix indices; value: the commutators for W matrix
            dict: key: a string of matrix indices; value: the commutators for M matrix
            dict: key: a string of matrix indices; value: the commutators for V matrix
            int: number of entries in the matrix
        """
        size = len(hopping_operators.keys())

        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops, untapered_op, z2_symmetries, sign):

            to_be_computed_list = []
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]
                left_op = available_hopping_ops.get('E_{}'.format(m_u))
                right_op_1 = available_hopping_ops.get('E_{}'.format(n_u))
                right_op_2 = available_hopping_ops.get('E_dag_{}'.format(n_u))
                to_be_computed_list.append((m_u, n_u, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(NumericalqEOMExcitedStatesCalculation._build_commutator_routine,
                                   to_be_computed_list,
                                   task_args=(untapered_op, z2_symmetries, sign),
                                   num_processes=aqua_globals.num_processes)
            for result in results:
                m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result

                all_matrix_operators['q_{}_{}'.format(m_u, n_u)] = q_mat_op
                all_matrix_operators['w_{}_{}'.format(m_u, n_u)] = w_mat_op
                all_matrix_operators['m_{}_{}'.format(m_u, n_u)] = m_mat_op
                all_matrix_operators['v_{}_{}'.format(m_u, n_u)] = v_mat_op

        try:
            z2_symmetries = self._gsc.transformation.z2_symmetries
        except:
            z2_symmetries = Z2Symmetries([],[],[])

        if not z2_symmetries.is_empty():
            for targeted_tapering_values in itertools.product([1, -1], repeat=len(z2_symmetries.symmetries)):

                logger.info("In sector: (%s)", ','.join([str(x) for x in targeted_tapering_values]))
                # remove the excited operators which are not suitable for the sector

                available_hopping_ops = {}
                targeted_sector = (np.asarray(targeted_tapering_values) == 1)
                for key, value in type_of_commutativities.items():
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = hopping_operators[key]
                _build_one_sector(available_hopping_ops, self._gsc.transformation.untapered_op,
                                  z2_symmetries, self._gcs.transormation.commutation_rule)

        else:
            _build_one_sector(hopping_operators,self._gsc.transformation.untapered_op,
                                  z2_symmetries, self._gcs.transormation.commutation_rule)


        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(params, operator, z2_symmetries, sign):
        m_u, n_u, left_op, right_op_1, right_op_2 = params
        if left_op is None:
            q_mat_op = None
            w_mat_op = None
            m_mat_op = None
            v_mat_op = None
        else:
            if right_op_1 is None and right_op_2 is None:
                q_mat_op = None
                w_mat_op = None
                m_mat_op = None
                v_mat_op = None
            else:
                if right_op_1 is not None:
                    q_mat_op = commutator(left_op, operator, right_op_1, sign=sign)
                    w_mat_op = commutator(left_op, right_op_1, sign = sign)
                    q_mat_op = None if q_mat_op.is_empty() else q_mat_op
                    w_mat_op = None if w_mat_op.is_empty() else w_mat_op
                else:
                    q_mat_op = None
                    w_mat_op = None

                if right_op_2 is not None:
                    m_mat_op = commutator(left_op, operator, right_op_2, sign = sign)
                    v_mat_op = commutator(left_op, right_op_2, sign = sign)
                    m_mat_op = None if m_mat_op.is_empty() else m_mat_op
                    v_mat_op = None if v_mat_op.is_empty() else v_mat_op
                else:
                    m_mat_op = None
                    v_mat_op = None

                if not z2_symmetries.is_empty():
                    if q_mat_op is not None and not q_mat_op.is_empty():
                        q_mat_op = z2_symmetries.taper(q_mat_op)
                    if w_mat_op is not None and not w_mat_op.is_empty():
                        w_mat_op = z2_symmetries.taper(w_mat_op)
                    if m_mat_op is not None and not m_mat_op.is_empty():
                        m_mat_op = z2_symmetries.taper(m_mat_op)
                    if v_mat_op is not None and not v_mat_op.is_empty():
                        v_mat_op = z2_symmetries.taper(v_mat_op)

        return m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op

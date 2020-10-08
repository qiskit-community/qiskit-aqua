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
from abc import abstractmethod
import logging
from scipy import linalg

from typing import Optional, List

from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.excited_states_calculation import ExcitedStatesCalculation

logger = logging.getLogger(__name__)


class qEOMExcitedStatesCalculation(ExcitedStatesCalculation):

    def __init__(self, ground_state_calculation: GroundStateCalculation,
                 excitation_type: str = 'SD', active_space: Optional[List[int]] = None):

        super().__init__(ground_state_calculation)

        self.excitation_type = excitation_type
        self.active_space = active_space

    @property
    def excitation_type(self) -> str:
        """Returns the type of excitations to be included in the eom pseudo-eignevalue problem."""
        return self._excitation_type

    @excitation_type.setter
    def excitation_type(self, excitation_type: str) -> None:
        """Sets the type of excitations to be included in the eom pseudo-eignevalue problem."""
        if [letter not in ['S','D','T'] for letter in excitation_type]:
            raise ValueError(
                'Excitation type must be S (single), D (double), T (triple) or a combination of them')
        self._excitation_type = excitation_type

    @property
    def active_space(self) -> Optional[List[int]]:
        """Returns the active space in which the excitations are defined."""
        return self._active_space

    @active_space.setter
    def active_space(self, active_space: List) -> None:
        """Sets the active space in which the excitations are defined."""
        self._active_space = active_space

    def compute_excitedstates(self, driver: BaseDriver):
        """
        construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients
        Args:
            driver:
        """

        # 1. Run ground state calculation
        ground_state_results = self._gsc.calculate_ground_state(driver)

        # 2. Prepare the excitation operators
        matrix_operators_dict = self.prepare_matrix_operators()

        # matrix_operators_names=[]
        # matrix_operators_paulis=[]
        # for name, pauli in matrix_operators_dict.items():
        #     if pauli:
        #         matrix_operators_names.append(name)
        #         matrix_operators_paulis.append(pauli)
        #
        # measurement_results = { name : ground_state_results.aux_values[idx]
        #                        for idx, name in enumerate(matrix_operators_names) }

        # 3. Evaluate eom operators
        measurement_results = self._gsc.evaluate_operators(matrix_operators_dict)

        # 3. Postprocess ground_state_result to construct eom matrices
        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self.build_eom_matrices(measurement_results)

        # 4. solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self.compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        results = {'excitation energies': energy_gaps,
                   'expansion coefficients': expansion_coefs,
                   'm_mat': m_mat, 'v_mat': v_mat, 'q_mat': q_mat, 'w_mat': w_mat,
                   'm_mat_std': m_mat_std, 'v_mat_std': v_mat_std,
                   'q_mat_std': q_mat_std, 'w_mat_std': w_mat_std}

        return results

    @abstractmethod
    def prepare_matrix_operators(self):
        raise NotImplementedError


    def build_eom_matrices(self, gs_results):
        """
        Constructs the M, V, Q and W matrices from the results on the ground state
        Args:
            gs_results:

        Returns:

        """

        size = len(gs_results.keys)

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0, 0, 0, 0

        # evaluate results
        for idx, _ in enumerate(mus):
            m_u = mus[idx]
            n_u = nus[idx]

            q_mat[m_u][n_u] = gs_results['q_{}_{}'.format(m_u, n_u)] if gs_results.get(
                'q_{}_{}'.format(m_u, n_u)) is not None else q_mat[m_u][n_u]
            w_mat[m_u][n_u] = gs_results['w_{}_{}'.format(m_u, n_u)] if gs_results.get(
                'w_{}_{}'.format(m_u, n_u)) is not None else w_mat[m_u][n_u]
            m_mat[m_u][n_u] = gs_results['m_{}_{}'.format(m_u, n_u)] if gs_results.get(
                'm_{}_{}'.format(m_u, n_u)) is not None else m_mat[m_u][n_u]
            v_mat[m_u][n_u] = gs_results['v_{}_{}'.format(m_u, n_u)] if gs_results.get(
                'v_{}_{}'.format(m_u, n_u)) is not None else v_mat[m_u][n_u]

            q_mat_std += gs_results['q_{}_{}_std'.format(m_u, n_u)] if gs_results.get(
                'q_{}_{}_std'.format(m_u, n_u)) is not None else 0
            w_mat_std += gs_results['w_{}_{}_std'.format(m_u, n_u)] if gs_results.get(
                'w_{}_{}_std'.format(m_u, n_u)) is not None else 0
            m_mat_std += gs_results['m_{}_{}_std'.format(m_u, n_u)] if gs_results.get(
                'm_{}_{}_std'.format(m_u, n_u)) is not None else 0
            v_mat_std += gs_results['v_{}_{}_std'.format(m_u, n_u)] if gs_results.get(
                'v_{}_{}_std'.format(m_u, n_u)) is not None else 0


        q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
        w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

        q_mat = np.real(q_mat)
        w_mat = np.real(w_mat)
        m_mat = np.real(m_mat)
        v_mat = np.real(v_mat)

        q_mat_std = q_mat_std / float(size**2)
        w_mat_std = w_mat_std / float(size**2)
        m_mat_std = m_mat_std / float(size**2)
        v_mat_std = v_mat_std / float(size**2)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nW:=========================\n%s", w_mat)
        logger.debug("\nM:=========================\n%s", m_mat)
        logger.debug("\nV:=========================\n%s", v_mat)

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std


    @staticmethod
    def compute_excitation_energies(m_mat, v_mat, q_mat, w_mat):
        """Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat (numpy.ndarray): M matrices
            v_mat (numpy.ndarray): V matrices
            q_mat (numpy.ndarray): Q matrices
            w_mat (numpy.ndarray): W matrices

        Returns:
            numpy.ndarray: 1-D vector stores all energy gap to reference state
            numpy.ndarray: 2-D array storing the X and Y expansion coefficients
        """
        logger.debug('Diagonalizing qeom matrices for excited states...')
        a_mat = np.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])
        # pylint: disable=too-many-function-args
        res = linalg.eig(a_mat, b_mat)
        # convert nan value into 0
        res[0][np.where(np.isnan(res[0]))] = 0.0
        # Only the positive eigenvalues are physical. We need to take care
        # though of very small values
        # should an excited state approach ground state. Here the small values
        # may be both negative or
        # positive. We should take just one of these pairs as zero. So to get the values we want we
        # sort the real parts and then take the upper half of the sorted values.
        # Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug('... %s', res[0])
        w = np.sort(np.real(res[0]))
        logger.debug('Sorted real parts %s', w)
        w = np.abs(w[len(w) // 2:])
        w[w < 1e-06] = 0
        excitation_energies_gap = w

        return excitation_energies_gap, res[1]
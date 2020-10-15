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

"""The calculation of excited states via the qEOM algorithm"""

from typing import List, Union, Optional
import logging
from abc import abstractmethod
import numpy as np
from scipy import linalg

from qiskit.aqua.algorithms import AlgorithmResult
from qiskit.chemistry import FermionicOperator, BosonicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.excited_states_calculation import ExcitedStatesCalculation
from qiskit.chemistry.results import EigenstateResult

logger = logging.getLogger(__name__)


class QEOMExcitedStatesCalculation(ExcitedStatesCalculation):
    """The calculation of excited states via the qEOM algorithm"""

    def __init__(self, ground_state_calculation: GroundStateCalculation,
                 excitations: Union[str, List[List[int]]] = 'sd'):
        """
        Args:
            ground_state_calculation: a GroundStateCalculation object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise a list of custom excitations can directly be provided.
        """

        self._gsc = ground_state_calculation
        self.excitations = excitations

    @property
    def excitations(self) -> Union[str, List[List[int]]]:
        """Returns the excitations to be included in the eom pseudo-eignevalue problem."""
        return self._excitations

    @excitations.setter
    def excitations(self, excitations: Union[str, List[List[int]]]) -> None:
        """The excitations to be included in the eom pseudo-eigenvalue problem. If a string then
        all excitations of given type will be used. Otherwise a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str) and any([letter not in ['s', 'd'] for letter in excitations]
                                                ):
            raise ValueError(
                'Excitation type must be s (singles), d (doubles) or sd (singles and doubles)')
        self._excitations = excitations

    def compute_excitedstates(self, driver: BaseDriver,
                              aux_operators: Optional[Union[List[FermionicOperator],
                                                            List[BosonicOperator]]] = None):
        """
        construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients
        Args:
            driver: a chemistry driver object which defines the chemical problem that is to be
                    solved by this calculation.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.
        """

        if aux_operators is not None:
            logger.warning("With qEOM the auxiliary operators can currently only be "
                           "evaluated on the ground state.")

        # 1. Run ground state calculation
        groundstate_result = self._gsc.compute_groundstate(driver, aux_operators)

        # 2. Prepare the excitation operators
        matrix_operators_dict, size = self._prepare_matrix_operators()

        # 3. Evaluate eom operators
        measurement_results = self._gsc.evaluate_operators(
            groundstate_result.raw_result['eigenstate'],
            matrix_operators_dict)

        # 4. Postprocess ground_state_result to construct eom matrices
        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self._build_eom_matrices(measurement_results, size)

        # 5. solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self._compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs
        qeom_result.excitation_energies = energy_gaps
        qeom_result.m_matrix = m_mat
        qeom_result.v_matrix = v_mat
        qeom_result.q_matrix = q_mat
        qeom_result.w_matrix = w_mat
        qeom_result.m_matrix_std = m_mat_std
        qeom_result.v_matrix_std = v_mat_std
        qeom_result.q_matrix_std = q_mat_std
        qeom_result.w_matrix_std = w_mat_std

        eigenstate_result = EigenstateResult()
        eigenstate_result.eigenstates = groundstate_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = groundstate_result.aux_operator_eigenvalues
        eigenstate_result.raw_result = qeom_result

        eigenstate_result.eigenenergies = np.append(groundstate_result.eigenenergies,
                                                    np.asarray([groundstate_result.eigenenergies[0]
                                                                + gap for gap in energy_gaps]))

        result = self._gsc.transformation.interpret(eigenstate_result)

        return result

    @abstractmethod
    def _prepare_matrix_operators(self):
        """construct the excitation operators for each matrix element"""
        raise NotImplementedError

    def _build_eom_matrices(self, gs_results: dict, size: int) -> [np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray, float,
                                                                   float, float, float]:
        """
        Constructs the M, V, Q and W matrices from the results on the ground state
        Args:
            size: size of eigenvalue problem
            gs_results: a ground state result object

        Returns: the matrices and their standard deviation

        """

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

            q_mat[m_u][n_u] = gs_results['q_{}_{}'.format(m_u, n_u)][0] if gs_results.get(
                'q_{}_{}'.format(m_u, n_u)) is not None else q_mat[m_u][n_u]
            w_mat[m_u][n_u] = gs_results['w_{}_{}'.format(m_u, n_u)][0] if gs_results.get(
                'w_{}_{}'.format(m_u, n_u)) is not None else w_mat[m_u][n_u]
            m_mat[m_u][n_u] = gs_results['m_{}_{}'.format(m_u, n_u)][0] if gs_results.get(
                'm_{}_{}'.format(m_u, n_u)) is not None else m_mat[m_u][n_u]
            v_mat[m_u][n_u] = gs_results['v_{}_{}'.format(m_u, n_u)][0] if gs_results.get(
                'v_{}_{}'.format(m_u, n_u)) is not None else v_mat[m_u][n_u]

            q_mat_std += gs_results['q_{}_{}_std'.format(m_u, n_u)][0] if gs_results.get(
                'q_{}_{}_std'.format(m_u, n_u)) is not None else 0
            w_mat_std += gs_results['w_{}_{}_std'.format(m_u, n_u)][0] if gs_results.get(
                'w_{}_{}_std'.format(m_u, n_u)) is not None else 0
            m_mat_std += gs_results['m_{}_{}_std'.format(m_u, n_u)][0] if gs_results.get(
                'm_{}_{}_std'.format(m_u, n_u)) is not None else 0
            v_mat_std += gs_results['v_{}_{}_std'.format(m_u, n_u)][0] if gs_results.get(
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
    def _compute_excitation_energies(m_mat: np.ndarray, v_mat: np.ndarray, q_mat: np.ndarray,
                                     w_mat: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat : M matrices
            v_mat : V matrices
            q_mat : Q matrices
            w_mat : W matrices

        Returns:
            1-D vector stores all energy gap to reference state
            2-D array storing the X and Y expansion coefficients
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


class QEOMResult(AlgorithmResult):

    @property
    def ground_state_raw_result(self):
        """ returns ground state raw result """
        return self.get('ground_state_raw_result')

    @ground_state_raw_result.setter
    def ground_state_raw_result(self, value) -> None:
        """ sets ground state raw result """
        self.data['ground_state_raw_result'] = value

    @property
    def excitation_energies(self) -> np.ndarray:
        """ returns the excitation energies (energy gaps) """
        return self.get('excitation_energies')

    @excitation_energies.setter
    def excitation_energies(self, value: np.ndarray) -> None:
        """ sets the excitation energies (energy gaps) """
        self.data['excitation_energies'] = value

    @property
    def expansion_coefficients(self) -> np.ndarray:
        """ returns the X and Y expansion coefficients """
        return self.get('expansion_coefficients')

    @expansion_coefficients.setter
    def expansion_coefficients(self, value: np.ndarray) -> None:
        """ sets the X and Y expansion coefficients """
        self.data['expansion_coefficients'] = value

    @property
    def m_matrix(self) -> np.ndarray:
        """ returns the M matrix """
        return self.get('m_matrix')

    @m_matrix.setter
    def m_matrix(self, value: np.ndarray) -> None:
        """ sets the M matrix """
        self.data['m_matrix'] = value

    @property
    def v_matrix(self) -> np.ndarray:
        """ returns the V matrix """
        return self.get('v_matrix')

    @v_matrix.setter
    def v_matrix(self, value: np.ndarray) -> None:
        """ sets the V matrix """
        self.data['v_matrix'] = value

    @property
    def q_matrix(self) -> np.ndarray:
        """ returns the Q matrix """
        return self.get('q_matrix')

    @q_matrix.setter
    def q_matrix(self, value: np.ndarray) -> None:
        """ sets the Q matrix """
        self.data['q_matrix'] = value

    @property
    def w_matrix(self) -> np.ndarray:
        """ returns the W matrix """
        return self.get('w_matrix')

    @w_matrix.setter
    def w_matrix(self, value: np.ndarray) -> None:
        """ sets the W matrix """
        self.data['w_matrix'] = value

    @property
    def m_matrix_std(self) -> float:
        """ returns the M matrix standard deviation """
        return self.get('m_matrix_std')

    @m_matrix_std.setter
    def m_matrix_std(self, value: float) -> None:
        """ sets the M matrix standard deviation """
        self.data['m_matrix_std'] = value

    @property
    def v_matrix_std(self) -> float:
        """ returns the V matrix standard deviation """
        return self.get('v_matrix_std')

    @v_matrix_std.setter
    def v_matrix_std(self, value: float) -> None:
        """ sets the V matrix standard deviation """
        self.data['v_matrix_std'] = value

    @property
    def q_matrix_std(self) -> float:
        """ returns the Q matrix standard deviation """
        return self.get('q_matrix_std')

    @q_matrix_std.setter
    def q_matrix_std(self, value: float) -> None:
        """ sets the Q matrix standard deviation """
        self.data['q_matrix_std'] = value

    @property
    def w_matrix_std(self) -> float:
        """ returns the W matrix standard deviation """
        return self.get('w_matrix_std')

    @w_matrix_std.setter
    def w_matrix_std(self, value: float) -> None:
        """ sets the W matrix standard deviation """
        self.data['w_matrix_std'] = value

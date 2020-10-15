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

"""The calculation of excited states via the qEOM algorithm"""

from typing import List, Union, Optional, Tuple, Dict, cast
import itertools
import logging
import sys
import numpy as np
from scipy import linalg

from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import AlgorithmResult
from qiskit.aqua.operators import Z2Symmetries, commutator, WeightedPauliOperator
from qiskit.chemistry import FermionicOperator, BosonicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import (ElectronicStructureResult, VibronicStructureResult,
                                      EigenstateResult)

from .excited_states_solver import ExcitedStatesSolver
from ..ground_state_solvers import GroundStateSolver

logger = logging.getLogger(__name__)


class QEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm"""

    def __init__(self, ground_state_solver: GroundStateSolver,
                 excitations: Union[str, List[List[int]]] = 'sd') -> None:
        """
        Args:
            ground_state_solver: a GroundStateSolver object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise a list of custom excitations can directly be provided.
        """
        self._gsc = ground_state_solver
        self.excitations = excitations

    @property
    def excitations(self) -> Union[str, List[List[int]]]:
        """Returns the excitations to be included in the eom pseudo-eigenvalue problem."""
        return self._excitations

    @excitations.setter
    def excitations(self, excitations: Union[str, List[List[int]]]) -> None:
        """The excitations to be included in the eom pseudo-eigenvalue problem. If a string then
        all excitations of given type will be used. Otherwise a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str):
            if excitations not in ['s', 'd', 'sd']:
                raise ValueError('Excitation type must be s (singles), d (doubles) or sd '
                                 '(singles and doubles)')
        self._excitations = excitations

    def solve(self, driver: BaseDriver,
              aux_operators: Optional[Union[List[FermionicOperator],
                                            List[BosonicOperator]]] = None
              ) -> Union[ElectronicStructureResult, VibronicStructureResult]:
        """Run the excited-states calculation.

        Construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            driver: a chemistry driver object which defines the chemical problem that is to be
                    solved by this calculation.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.

        Returns:
            The excited states result. In case of a fermionic problem a
            ``ElectronicStructureResult`` is returned and in the bosonic case a
            ``VibronicStructureResult``.
        """

        if aux_operators is not None:
            logger.warning("With qEOM the auxiliary operators can currently only be "
                           "evaluated on the ground state.")

        # 1. Run ground state calculation
        groundstate_result = self._gsc.solve(driver, aux_operators)

        # 2. Prepare the excitation operators
        matrix_operators_dict, size = self._prepare_matrix_operators()

        # 3. Evaluate eom operators
        measurement_results = self._gsc.evaluate_operators(
            groundstate_result.raw_result['eigenstate'],
            matrix_operators_dict)
        measurement_results = cast(Dict[str, List[float]], measurement_results)

        # 4. Post-process ground_state_result to construct eom matrices
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

    def _prepare_matrix_operators(self) -> Tuple[dict, int]:
        """Construct the excitation operators for each matrix element.

        Returns:
            a dictionary of all matrix elements operators and the number of excitations
            (or the size of the qEOM pseudo-eigenvalue problem)
        """
        data = self._gsc.transformation.build_hopping_operators(self._excitations)
        hopping_operators, type_of_commutativities, excitation_indices = data

        size = int(len(list(excitation_indices.keys()))/2)

        eom_matrix_operators = self._build_all_commutators(
            hopping_operators, type_of_commutativities, size)

        return eom_matrix_operators, size

    def _build_all_commutators(self, hopping_operators: dict, type_of_commutativities: dict,
                               size: int) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            hopping_operators: all hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.
            type_of_commutativities: if tapering is used, it records the commutativities of
                hopping operators with the
                Z2 symmetries found in the original operator.
            size: the number of excitations (size of the qEOM pseudo-eigenvalue problem)

        Returns:
            a dictionary that contains the operators for each matrix element
        """

        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops, untapered_op, z2_symmetries, sign):

            to_be_computed_list = []
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]
                left_op = available_hopping_ops.get('E_{}'.format(m_u))
                right_op_1 = available_hopping_ops.get('E_{}'.format(n_u))
                right_op_2 = available_hopping_ops.get('Edag_{}'.format(n_u))
                to_be_computed_list.append((m_u, n_u, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(self._build_commutator_routine,
                                   to_be_computed_list,
                                   task_args=(untapered_op, z2_symmetries, sign),
                                   num_processes=aqua_globals.num_processes)
            for result in results:
                m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result

                if q_mat_op is not None:
                    all_matrix_operators['q_{}_{}'.format(m_u, n_u)] = q_mat_op
                if w_mat_op is not None:
                    all_matrix_operators['w_{}_{}'.format(m_u, n_u)] = w_mat_op
                if m_mat_op is not None:
                    all_matrix_operators['m_{}_{}'.format(m_u, n_u)] = m_mat_op
                if v_mat_op is not None:
                    all_matrix_operators['v_{}_{}'.format(m_u, n_u)] = v_mat_op

        try:
            # The next step only works in the case of the FermionicTransformation. Thus, it is done
            # in a try-except block. However, mypy doesn't detect this and thus we ignore it.
            z2_symmetries = self._gsc.transformation.molecule_info['z2_symmetries']  # type: ignore
        except AttributeError:
            z2_symmetries = Z2Symmetries([], [], [])

        if not z2_symmetries.is_empty():
            combinations = itertools.product([1, -1], repeat=len(z2_symmetries.symmetries))
            for targeted_tapering_values in combinations:
                logger.info("In sector: (%s)", ','.join([str(x) for x in targeted_tapering_values]))
                # remove the excited operators which are not suitable for the sector

                available_hopping_ops = {}
                targeted_sector = (np.asarray(targeted_tapering_values) == 1)
                for key, value in type_of_commutativities.items():
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = hopping_operators[key]
                # untapered_qubit_op is a WeightedPauliOperator and should not be exposed.
                _build_one_sector(available_hopping_ops,
                                  self._gsc.transformation.untapered_qubit_op,  # type: ignore
                                  z2_symmetries,
                                  self._gsc.transformation.commutation_rule)

        else:
            # untapered_qubit_op is a WeightedPauliOperator and should not be exposed.
            _build_one_sector(hopping_operators,
                              self._gsc.transformation.untapered_qubit_op,  # type: ignore
                              z2_symmetries,
                              self._gsc.transformation.commutation_rule)

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(params: List, operator: WeightedPauliOperator,
                                  z2_symmetries: Z2Symmetries, sign: int
                                  ) -> Tuple[int, int, WeightedPauliOperator, WeightedPauliOperator,
                                             WeightedPauliOperator, WeightedPauliOperator]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators
            operator: the hamiltonian
            z2_symmetries: z2_symmetries in case of tapering
            sign: commute or anticommute

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices
        """
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
                    w_mat_op = commutator(left_op, right_op_1, sign=sign)
                    q_mat_op = None if q_mat_op.is_empty() else q_mat_op
                    w_mat_op = None if w_mat_op.is_empty() else w_mat_op
                else:
                    q_mat_op = None
                    w_mat_op = None

                if right_op_2 is not None:
                    m_mat_op = commutator(left_op, operator, right_op_2, sign=sign)
                    v_mat_op = commutator(left_op, right_op_2, sign=sign)
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

    def _build_eom_matrices(self, gs_results: Dict[str, List[float]], size: int
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       float, float, float, float]:
        """Constructs the M, V, Q and W matrices from the results on the ground state

        Args:
            gs_results: a ground state result object
            size: size of eigenvalue problem

        Returns:
            the matrices and their standard deviation
        """

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0., 0., 0., 0.

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

        # these matrices are numpy arrays and therefore have the ``shape`` attribute
        # pylint: disable=unsubscriptable-object
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
                                     w_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    """The results class for the QEOM algorithm."""

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

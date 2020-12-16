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
import numpy as np

from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry import FermionicOperator, BosonicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import (ElectronicStructureResult, VibronicStructureResult,
                                      EigenstateResult)

from .qeom import QEOM, QEOMResult
from ..ground_state_solvers import GroundStateSolver
from .analytic_qeom_utils import (commutator_adj_nor,
                                  commutator_adj_adj,
                                  triple_commutator_adj_twobody_nor,
                                  triple_commutator_adj_twobody_adj)

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class AnalyticQEOM(QEOM):
    """ Analytic QEOM """

    def __init__(self, ground_state_solver: GroundStateSolver,
                 excitations: Union[str, List[List[int]]] = 'sd') -> None:
        super().__init__(ground_state_solver, excitations)
        try:
            from mpi4py import MPI
            self.size = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.name = MPI.Get_processor_name()
            self.comm = MPI.COMM_WORLD
            self.ispar = True
        except ImportError:
            self.size = 1
            self.rank = 0
            self.name = 'root'
            self.comm = None
            self.ispar = False

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

        print("ANALYTICAL SOLVER")

        if aux_operators is not None:
            logger.warning("With qEOM the auxiliary operators can currently only be "
                           "evaluated on the ground state.")

        # 1. Run ground state calculation
        groundstate_result = self._gsc.solve(driver, aux_operators)

        # synchronize nodes
        if self.ispar:
            groundstate_result = self.comm.bcast(groundstate_result, root=0)

        # 2. Prepare the excitation operators
        matrix_operators_dict, size = self._prepare_matrix_operators(driver)

        # 3. Evaluate eom operators
        measurement_results = self._gsc.evaluate_operators(
            groundstate_result.raw_result['eigenstate'],
            matrix_operators_dict)
        measurement_results = cast(Dict[str, List[float]], measurement_results)

        if self.ispar:
            res = self.comm.gather(measurement_results, root=0)
            res = self.comm.bcast(res, root=0)
            measurement_results = {}
            for item in res:
                measurement_results.update(item)
        for k in measurement_results.keys():
            print("measurements ", k, measurement_results[k])

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

    # pylint: disable=arguments-differ
    def _prepare_matrix_operators(self, driver) -> Tuple[dict, int]:  # type: ignore
        data = self._gsc.transformation.build_hopping_operators(self._excitations)
        _, _, excitation_indices = data

        for k in excitation_indices.keys():
            if len(excitation_indices[k]) == 2:
                i, a = excitation_indices[k]
                excitation_indices[k] = [a, i]
            if len(excitation_indices[k]) == 4:
                i, a, j, b = excitation_indices[k]
                excitation_indices[k] = [a, b, i, j]

        size = int(len(list(excitation_indices.keys()))/2)

        matrix_elements = [(I, J) for I, J in itertools.product(range(size), repeat=2) if I <= J]
        matrix_elements = \
            [matrix_elements[k] for k in range(len(matrix_elements)) if k % self.size == self.rank]

        n, h = self._prepare_hamiltonian(driver)

        eom_matrix_operators = {}
        for I, J in matrix_elements:
            EI = excitation_indices['E_%d' % I]
            EJ = excitation_indices['E_%d' % J]
            # print("EXC ", EI, EJ)
            adj_nor_oper, adj_nor_idx = commutator_adj_nor(n, EI, EJ)  # ; print("V")
            eom_matrix_operators['v_%d_%d' % (I, J)] = self.transform(adj_nor_oper, adj_nor_idx)
            adj_adj_oper, adj_adj_idx = commutator_adj_adj(n, EI, EJ)  # ; print("W")
            eom_matrix_operators['w_%d_%d' % (I, J)] = -1*self.transform(adj_adj_oper, adj_adj_idx)
            adj_nor_oper, adj_nor_idx = \
                triple_commutator_adj_twobody_nor(n, EI, EJ, h)  # ; print("M")
            eom_matrix_operators['m_%d_%d' % (I, J)] = 0.5*self.transform(adj_nor_oper, adj_nor_idx)
            adj_adj_oper, adj_adj_idx = \
                triple_commutator_adj_twobody_adj(n, EI, EJ, h)  # ; print("Q")
            eom_matrix_operators['q_%d_%d' % (I, J)] = 0.5*self.transform(adj_adj_oper, adj_adj_idx)
        return eom_matrix_operators, size

    def _prepare_hamiltonian(self, driver):
        q_molecule = driver.run()
        h1 = q_molecule.mo_onee_ints
        h2 = q_molecule.mo_eri_ints
        n = h1.shape[0]
        core_list = q_molecule.core_orbitals if self._gsc.transformation._freeze_core else []
        reduce_list = core_list + self._gsc.transformation._orbital_reduction
        occ_reduction = \
            [x for x in reduce_list if x < min(q_molecule.num_alpha, q_molecule.num_beta)]
        vrt_reduction = \
            [x for x in reduce_list if x >= max(q_molecule.num_alpha, q_molecule.num_beta)]
        nptot = q_molecule.num_alpha+q_molecule.num_beta-2*len(occ_reduction)
        to_save = [x for x in range(n) if x not in occ_reduction and x not in vrt_reduction]
        for i in occ_reduction:
            h1 += 2*h2[i, i, :, :] - h2[i, :, :, i]
        h1 = h1[np.ix_(to_save, to_save)]
        h2 = h2[np.ix_(to_save, to_save, to_save, to_save)]
        n = h1.shape[0]
        t1 = np.zeros((2*n, 2*n))
        t1[:n, :n] = h1
        t1[n:, n:] = h1
        h = np.zeros((2*n, 2*n, 2*n, 2*n))
        h = np.einsum('pr,qs->prqs', t1, np.eye(2*n)/(nptot-1.0))
        h[:n, :n, :n, :n] += 0.5*h2
        h[:n, :n, n:, n:] += 0.5*h2
        h[n:, n:, :n, :n] += 0.5*h2
        h[n:, n:, n:, n:] += 0.5*h2
        return 2*n, h

    def transform(self, oper, idx):
        """ transform """
        qubit_mapping = self._gsc.transformation._qubit_mapping
        two_qubit_reduction = self._gsc.transformation._two_qubit_reduction
        num_particles = self._gsc.transformation._molecule_info['num_particles']
        epsilon = 1e-8
        oper = oper.mapping(map_type=qubit_mapping, threshold=epsilon, idx=idx)
        if qubit_mapping == 'parity' and two_qubit_reduction:
            oper = Z2Symmetries.two_qubit_reduction(oper, num_particles)
        # print("operator ", oper.print_details())
        return oper

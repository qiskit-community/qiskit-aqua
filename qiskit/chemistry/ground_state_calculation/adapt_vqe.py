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
A ground state calculation employing the AdaptVQE algorithm.
"""

from typing import Optional, List
import logging
import re
import warnings
import numpy as np

from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQEResult, VQE
from qiskit.aqua.operators import LegacyBaseOperator, WeightedPauliOperator
from qiskit.aqua.utils.validation import validate_min
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.qubit_transformations import FermionicTransformation
from qiskit.chemistry.results import FermionicGroundStateResult

from .ground_state_calculation import GroundStateCalculation
from .mes_factories import VQEUCCSDFactory

logger = logging.getLogger(__name__)


class AdaptVQE(GroundStateCalculation):
    """A ground state calculation employing the AdaptVQE algorithm."""

    def __init__(self,
                 transformation: FermionicTransformation,
                 solver: VQEUCCSDFactory,
                 threshold: float = 1e-5,
                 delta: float = 1,
                 max_iterations: Optional[int] = None,
                 ) -> None:
        """
        Args:
            transformation: a fermionic driver to operator transformation strategy.
            solver: a factory for the VQE solver employing a UCCSD variational form.
            threshold: the energy convergence threshold. It has a minimum value of 1e-15.
            delta: the finite difference step size for the gradient computation. It has a minimum
                   value of 1e-5.
            max_iterations: the maximum number of iterations of the AdaptVQE algorithm.
        """
        validate_min('threshold', threshold, 1e-15)
        validate_min('delta', delta, 1e-5)

        super().__init__(transformation)

        self._solver = solver
        self._threshold = threshold
        self._delta = delta
        self._max_iterations = max_iterations

    def returns_groundstate(self) -> bool:
        return True

    def _compute_gradients(self,
                           excitation_pool: List[WeightedPauliOperator],
                           theta: List[float],
                           var_form: UCCSD,
                           operator: LegacyBaseOperator,
                           ) -> List:
        """
        Computes the gradients for all available excitation operators.

        Args:
            excitation_pool: pool of excitation operators
            theta: list of (up to now) optimal parameters
            var_form: current variational form
            operator: system Hamiltonian

        Returns:
            List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        for exc in excitation_pool:
            # push next excitation to variational form
            var_form.push_hopping_operator(exc)
            # construct auxiliary VQE instance
            vqe = self._solver.get_solver(self._transformation)
            vqe.operator = operator
            vqe.var_form = var_form
            vqe.initial_point = var_form.preferred_init_points
            # evaluate energies
            parameter_sets = theta + [-self._delta] + theta + [self._delta]
            energy_results = vqe._energy_evaluation(np.asarray(parameter_sets))
            # compute gradient
            gradient = (energy_results[0] - energy_results[1]) / (2 * self._delta)
            res.append((np.abs(gradient), exc))
            # pop excitation from variational form
            var_form.pop_hopping_operator()

        return res

    @staticmethod
    def _check_cyclicity(indices: List[int]) -> bool:
        """
        Auxiliary function to check for cycles in the indices of the selected excitations.

        Args:
            indices: the list of chosen gradient indices.
        Returns:
            Whether repeating sequences of indices have been detected.
        """
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        # reg-ex explanation:
        # 1. (\b.+ .+\b) will match at least two numbers and try to match as many as possible. The
        #    word boundaries in the beginning and end ensure that now numbers are split into digits.
        # 2. the match of this part is placed into capture group 1
        # 3. ( \b\1\b)+ will match a space followed by the contents of capture group 1 (again
        #    delimited by word boundaries to avoid separation into digits).
        # -> this results in any sequence of at least two numbers being detected
        match = cycle_regex.search(' '.join(map(str, indices)))
        logger.debug('Cycle detected: %s', match)
        # Additionally we also need to check whether the last two numbers are identical, because the
        # reg-ex above will only find cycles of at least two consecutive numbers.
        # It is sufficient to assert that the last two numbers are different due to the iterative
        # nature of the algorithm.
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    def compute_groundstate(self, driver: BaseDriver) -> FermionicGroundStateResult:
        """Computes the ground state.

        Args:
            driver: a chemistry driver.
        Raises:
            AquaError: if a solver other than VQE or a variational form other than UCCSD is provided
                       or if the algorithm finishes due to an unforeseen reason.
        Returns:
            A fermionic ground state result.
        """
        operator, aux_operators = self._transformation.transform(driver)

        vqe = self._solver.get_solver(self._transformation)
        if not isinstance(vqe, VQE):
            raise AquaError("The AdaptVQE algorithm requires the use of the VQE solver")
        var_form = vqe.var_form
        if not isinstance(var_form, UCCSD):
            raise AquaError("The AdaptVQE algorithm requires the use of the UCCSD variational form")

        var_form.manage_hopping_operators()
        excitation_pool = var_form.excitation_pool

        threshold_satisfied = False
        alternating_sequence = False
        max_iterations_exceeded = False
        prev_op_indices = []
        theta = []  # type: List
        max_grad = (0, 0)
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info('--- Iteration #%s ---', str(iteration))
            # compute gradients
            cur_grads = self._compute_gradients(excitation_pool, theta, var_form, operator)
            # pick maximum gradient
            max_grad_index, max_grad = max(enumerate(cur_grads),
                                           key=lambda item: np.abs(item[1][0]))
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            gradlog = "\nGradients in iteration #{}".format(str(iteration))
            gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
            for i, grad in enumerate(cur_grads):
                gradlog += '\n{}: {}: {}'.format(str(i), str(grad[1]), str(grad[0]))
                if grad[1] == max_grad[1]:
                    gradlog += '\t(*)'
            logger.info(gradlog)
            if np.abs(max_grad[0]) < self._threshold:
                logger.info("Adaptive VQE terminated succesfully with a final maximum gradient: %s",
                            str(np.abs(max_grad[0])))
                threshold_satisfied = True
                break
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Alternating sequence found. Finishing.")
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                alternating_sequence = True
                break
            # add new excitation to self._var_form
            var_form.push_hopping_operator(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            vqe.var_form = var_form
            vqe.initial_point = theta
            raw_vqe_result = vqe.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
        else:
            # reached maximum number of iterations
            max_iterations_exceeded = True
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None and aux_operators:
            vqe.compute_minimum_eigenvalue(operator, aux_operators)

        if threshold_satisfied:
            finishing_criterion = 'Threshold converged'
        elif alternating_sequence:
            finishing_criterion = 'Aborted due to cyclicity'
        elif max_iterations_exceeded:
            finishing_criterion = 'Maximum number of iterations reached'
        else:
            raise AquaError('The algorithm finished due to an unforeseen reason!')

        # extend VQE returned information with additional outputs
        raw_result = AdaptVQEResult()
        raw_result.combine(raw_vqe_result)
        raw_result.num_iterations = iteration
        raw_result.final_max_gradient = max_grad[0]
        raw_result.finishing_criterion = finishing_criterion

        logger.info('The final energy is: %s', str(raw_result.optimal_value.real))
        return self.transformation.interpret(raw_result.eigenvalue, raw_result.eigenstate,
                                             raw_result.aux_operator_eigenvalues)


class AdaptVQEResult(VQEResult):
    """ AdaptVQE Result."""

    @property
    def num_iterations(self) -> int:
        """ Returns number of iterations """
        return self.get('num_iterations')

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """ Sets number of iterations """
        self.data['num_iterations'] = value

    @property
    def final_max_gradient(self) -> float:
        """ Returns final maximum gradient """
        return self.get('final_max_gradient')

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """ Sets final maximum gradient """
        self.data['final_max_gradient'] = value

    @property
    def finishing_criterion(self) -> str:
        """ Returns finishing criterion """
        return self.get('finishing_criterion')

    @finishing_criterion.setter
    def finishing_criterion(self, value: str) -> None:
        """ Sets finishing criterion """
        self.data['finishing_criterion'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'final_max_grad':
            warnings.warn('final_max_grad deprecated, use final_max_gradient property.',
                          DeprecationWarning)
            return super().__getitem__('final_max_gradient')

        return super().__getitem__(key)

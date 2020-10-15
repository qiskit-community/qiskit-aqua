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

"""A ground state calculation employing the AdaptVQE algorithm."""

import re
import logging
from typing import Optional, List, Tuple, Union
import numpy as np

from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import VQE
from qiskit.aqua import AquaError
from ...results.electronic_structure_result import ElectronicStructureResult
from ...results.vibronic_structure_result import VibronicStructureResult
from ...transformations.fermionic_transformation import FermionicTransformation
from ...drivers.base_driver import BaseDriver
from ...components.variational_forms import UCCSD
from ...fermionic_operator import FermionicOperator
from ...bosonic_operator import BosonicOperator

from .minimum_eigensolver_factories import MinimumEigensolverFactory
from .ground_state_eigensolver import GroundStateEigensolver

logger = logging.getLogger(__name__)


class AdaptVQE(GroundStateEigensolver):
    """A ground state calculation employing the AdaptVQE algorithm."""

    def __init__(self,
                 transformation: FermionicTransformation,
                 solver: MinimumEigensolverFactory,
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

        super().__init__(transformation, solver)

        self._threshold = threshold
        self._delta = delta
        self._max_iterations = max_iterations

    def returns_groundstate(self) -> bool:
        return True

    def _compute_gradients(self,
                           excitation_pool: List[WeightedPauliOperator],
                           theta: List[float],
                           vqe: VQE,
                           ) -> List[Tuple[float, WeightedPauliOperator]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            excitation_pool: pool of excitation operators
            theta: list of (up to now) optimal parameters
            vqe: the variational quantum eigensolver instance used for solving

        Returns:
            List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        for exc in excitation_pool:
            # push next excitation to variational form
            vqe.var_form.push_hopping_operator(exc)
            # NOTE: because we overwrite the var_form inside of the VQE, we need to update the VQE's
            # internal _var_form_params, too. We can do this by triggering the var_form setter. Once
            # the VQE does not store this pure var_form property any longer this can be removed.
            vqe.var_form = vqe.var_form
            # We also need to invalidate the internally stored expectation operator because it needs
            # to be updated for the new var_form.
            vqe._expect_op = None
            # evaluate energies
            parameter_sets = theta + [-self._delta] + theta + [self._delta]
            energy_results = vqe._energy_evaluation(np.asarray(parameter_sets))
            # compute gradient
            gradient = (energy_results[0] - energy_results[1]) / (2 * self._delta)
            res.append((np.abs(gradient), exc))
            # pop excitation from variational form
            vqe.var_form.pop_hopping_operator()

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

    def solve(self,
              driver: BaseDriver,
              aux_operators: Optional[Union[List[FermionicOperator],
                                            List[BosonicOperator]]] = None) \
            -> Union[ElectronicStructureResult, VibronicStructureResult]:

        """Computes the ground state.

        Args:
            driver: a chemistry driver.
            aux_operators: Additional auxiliary ``FermionicOperator``s to evaluate at the
                ground state.

        Raises:
            AquaError: if a solver other than VQE or a variational form other than UCCSD is provided
                       or if the algorithm finishes due to an unforeseen reason.

        Returns:
            An AdaptVQEResult which is an ElectronicStructureResult but also includes runtime
            information about the AdaptVQE algorithm like the number of iterations, finishing
            criterion, and the final maximum gradient.
        """
        operator, aux_operators = self._transformation.transform(driver, aux_operators)

        vqe = self._solver.get_solver(self._transformation)
        vqe.operator = operator
        if not isinstance(vqe, VQE):
            raise AquaError("The AdaptVQE algorithm requires the use of the VQE solver")
        if not isinstance(vqe.var_form, UCCSD):
            raise AquaError("The AdaptVQE algorithm requires the use of the UCCSD variational form")

        vqe.var_form.manage_hopping_operators()
        excitation_pool = vqe.var_form.excitation_pool

        threshold_satisfied = False
        alternating_sequence = False
        max_iterations_exceeded = False
        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[WeightedPauliOperator]] = (0., None)
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info('--- Iteration #%s ---', str(iteration))
            # compute gradients

            cur_grads = self._compute_gradients(excitation_pool, theta, vqe)
            # pick maximum gradient
            max_grad_index, max_grad = max(enumerate(cur_grads),
                                           key=lambda item: np.abs(item[1][0]))
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            if logger.isEnabledFor(logging.INFO):
                gradlog = "\nGradients in iteration #{}".format(str(iteration))
                gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
                for i, grad in enumerate(cur_grads):
                    gradlog += '\n{}: {}: {}'.format(str(i), str(grad[1]), str(grad[0]))
                    if grad[1] == max_grad[1]:
                        gradlog += '\t(*)'
                logger.info(gradlog)
            if np.abs(max_grad[0]) < self._threshold:
                logger.info("Adaptive VQE terminated successfully "
                            "with a final maximum gradient: %s",
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
            vqe.var_form.push_hopping_operator(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            vqe.initial_point = theta
            raw_vqe_result = vqe.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
        else:
            # reached maximum number of iterations
            max_iterations_exceeded = True
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            aux_values = self.evaluate_operators(raw_vqe_result.eigenstate, aux_operators)
        else:
            aux_values = None
        raw_vqe_result.aux_operator_eigenvalues = aux_values

        if threshold_satisfied:
            finishing_criterion = 'Threshold converged'
        elif alternating_sequence:
            finishing_criterion = 'Aborted due to cyclicity'
        elif max_iterations_exceeded:
            finishing_criterion = 'Maximum number of iterations reached'
        else:
            raise AquaError('The algorithm finished due to an unforeseen reason!')

        electronic_result = self.transformation.interpret(raw_vqe_result)

        result = AdaptVQEResult(electronic_result.data)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.finishing_criterion = finishing_criterion

        logger.info('The final energy is: %s', str(result.computed_electronic_energy))
        return result


class AdaptVQEResult(ElectronicStructureResult):
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

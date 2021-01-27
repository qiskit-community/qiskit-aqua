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

"""An implementation of the ADMM algorithm."""
import copy
import logging
import time
import warnings
from typing import List, Optional, Tuple, cast

import numpy as np
from qiskit.aqua.algorithms import NumPyMinimumEigensolver

from .minimum_eigen_optimizer import MinimumEigenOptimizer
from .optimization_algorithm import (OptimizationResultStatus, OptimizationAlgorithm,
                                     OptimizationResult)
from .slsqp_optimizer import SlsqpOptimizer
from ..problems.constraint import Constraint
from ..problems.linear_constraint import LinearConstraint
from ..problems.quadratic_objective import QuadraticObjective
from ..problems.quadratic_program import QuadraticProgram
from ..problems.variable import VarType, Variable

UPDATE_RHO_BY_TEN_PERCENT = 0
UPDATE_RHO_BY_RESIDUALS = 1

logger = logging.getLogger(__name__)


class ADMMParameters:
    """Defines a set of parameters for ADMM optimizer."""

    def __init__(self,
                 rho_initial: float = 10000,
                 factor_c: float = 100000,
                 beta: float = 1000,
                 maxiter: int = 10,
                 tol: float = 1.e-4,
                 max_time: float = np.inf,
                 three_block: bool = True,
                 vary_rho: int = UPDATE_RHO_BY_TEN_PERCENT,
                 tau_incr: float = 2,
                 tau_decr: float = 2,
                 mu_res: float = 10,
                 mu_merit: float = 1000,
                 warm_start: bool = False,
                 max_iter: Optional[int] = None) -> None:
        """Defines parameters for ADMM optimizer and their default values.

        Args:
            rho_initial: Initial value of rho parameter of ADMM.
            factor_c: Penalizing factor for equality constraints, when mapping to QUBO.
            beta: Penalization for y decision variables.
            maxiter: Maximum number of iterations for ADMM.
            tol: Tolerance for the residual convergence.
            max_time: Maximum running time (in seconds) for ADMM.
            three_block: Boolean flag to select the 3-block ADMM implementation.
            vary_rho: Flag to select the rule to update rho.
                If set to 0, then rho increases by 10% at each iteration.
                If set to 1, then rho is modified according to primal and dual residuals.
            tau_incr: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
                The update rule can be found in:
                Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
                Distributed optimization and statistical learning via the alternating
                direction method of multipliers.
                Foundations and Trends® in Machine learning, 3(1), 1-122.
            tau_decr: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
            mu_res: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
            mu_merit: Penalization for constraint residual. Used to compute the merit values.
            warm_start: Start ADMM with pre-initialized values for binary and continuous variables
                by solving a relaxed (all variables are continuous) problem first. This option does
                not guarantee the solution will optimal or even feasible. The option should be
                used when tuning other options does not help and should be considered as a hint
                to the optimizer where to start its iterative process.
            max_iter: Deprecated, use maxiter.
        """
        super().__init__()
        if max_iter is not None:
            warnings.warn('The max_iter parameter is deprecated as of '
                          '0.8.0 and will be removed no sooner than 3 months after the release. '
                          'You should use maxiter instead.',
                          DeprecationWarning)
            maxiter = max_iter
        self.mu_merit = mu_merit
        self.mu_res = mu_res
        self.tau_decr = tau_decr
        self.tau_incr = tau_incr
        self.vary_rho = vary_rho
        self.three_block = three_block
        self.max_time = max_time
        self.tol = tol
        self.maxiter = maxiter
        self.factor_c = factor_c
        self.beta = beta
        self.rho_initial = rho_initial
        self.warm_start = warm_start

    def __repr__(self) -> str:
        props = ", ".join(["{}={}".format(key, value) for (key, value) in vars(self).items()])
        return "{0}({1})".format(type(self).__name__, props)


class ADMMState:
    """Internal computation state of the ADMM implementation.

    The state keeps track of various variables are stored that are being updated during problem
    solving. The values are relevant to the problem being solved. The state is recreated for each
    optimization problem. State is returned as the third value.
    """

    def __init__(self,
                 op: QuadraticProgram,
                 rho_initial: float) -> None:
        """
        Args:
            op: The optimization problem being solved.
            rho_initial: Initial value of the rho parameter.
        """
        super().__init__()

        # Optimization problem itself
        self.op = op
        # Indices of the variables
        self.binary_indices = None  # type: Optional[List[int]]
        self.continuous_indices = None  # type: Optional[List[int]]
        self.step1_absolute_indices = None  # type: Optional[List[int]]
        self.step1_relative_indices = None  # type: Optional[List[int]]

        # define heavily used matrix, they are used at each iteration, so let's cache them,
        # they are np.ndarrays
        # pylint:disable=invalid-name
        # objective
        self.q0 = None  # type: Optional[np.ndarray]
        self.c0 = None  # type: Optional[np.ndarray]
        self.q1 = None  # type: Optional[np.ndarray]
        self.c1 = None  # type: Optional[np.ndarray]
        # constraints
        self.a0 = None  # type: Optional[np.ndarray]
        self.b0 = None  # type: Optional[np.ndarray]

        # These are the parameters that are updated in the ADMM iterations.
        self.u = np.zeros(op.get_num_continuous_vars())
        binary_size = op.get_num_binary_vars()
        self.x0 = np.zeros(binary_size)
        self.z = np.zeros(binary_size)
        self.z_init = self.z
        self.y = np.zeros(binary_size)
        self.lambda_mult = np.zeros(binary_size)

        # The following structures store quantities obtained in each ADMM iteration.
        self.cost_iterates = []  # type: List[float]
        self.residuals = []  # type: List[float]
        self.dual_residuals = []  # type: List[float]
        self.cons_r = []  # type: List[float]
        self.merits = []  # type: List[float]
        self.lambdas = []  # type: List[float]
        self.x0_saved = []  # type: List[np.ndarray]
        self.u_saved = []  # type: List[np.ndarray]
        self.z_saved = []  # type: List[np.ndarray]
        self.y_saved = []  # type: List[np.ndarray]
        self.rho = rho_initial

        # lin. eq. constraints with bin. vars. only
        self.binary_equality_constraints = []  # type: List[LinearConstraint]
        # all equality constraints
        self.equality_constraints = []  # type: List[Constraint]
        # all inequality constraints
        self.inequality_constraints = []  # type: List[Constraint]


class ADMMOptimizationResult(OptimizationResult):
    """ ADMMOptimization Result."""

    def __init__(self, x: np.ndarray, fval: float, variables: List[Variable],
                 state: ADMMState, status: OptimizationResultStatus) -> None:
        """
        Args:
            x: the optimal value found by ADMM.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            state: the internal computation state of ADMM.
            status: Termination status of an optimization algorithm
        """
        super().__init__(x=x, fval=fval, variables=variables, status=status, raw_results=state)

    @property
    def state(self) -> ADMMState:
        """ returns state """
        return self._raw_results


class ADMMOptimizer(OptimizationAlgorithm):
    """An implementation of the ADMM-based heuristic.

    This algorithm is introduced in [1].

    **References:**

    [1] Gambella, C., & Simonetto, A. (2020). Multi-block ADMM Heuristics for Mixed-Binary
        Optimization on Classical and Quantum Computers. arXiv preprint arXiv:2001.02069.
    """

    def __init__(self, qubo_optimizer: Optional[OptimizationAlgorithm] = None,
                 continuous_optimizer: Optional[OptimizationAlgorithm] = None,
                 params: Optional[ADMMParameters] = None) -> None:
        """
        Args:
            qubo_optimizer: An instance of OptimizationAlgorithm that can effectively solve
                QUBO problems. If not specified then :class:`MinimumEigenOptimizer` initialized
                with an instance of :class:`NumPyMinimumEigensolver` will be used.
            continuous_optimizer: An instance of OptimizationAlgorithm that can solve
                continuous problems. If not specified then :class:`SlsqpOptimizer` will be used.
            params: An instance of ADMMParameters.
        """
        super().__init__()
        self._log = logging.getLogger(__name__)

        # create default params if not present
        self._params = params or ADMMParameters()

        # create optimizers if not specified
        self._qubo_optimizer = qubo_optimizer or MinimumEigenOptimizer(NumPyMinimumEigensolver())
        self._continuous_optimizer = continuous_optimizer or SlsqpOptimizer()

        # internal state where we'll keep intermediate solution
        # here, we just declare the class variable, the variable is initialized in kept in
        # the solve method.
        self._state = None  # type: Optional[ADMMState]

    def get_compatibility_msg(self, problem: QuadraticProgram) -> Optional[str]:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, otherwise raises an error.

        Raises:
            QiskitOptimizationError: If the problem is not compatible with the ADMM optimizer.
        """

        msg = ''

        # 1. get bin/int and continuous variable indices
        bin_int_indices = self._get_variable_indices(problem, Variable.Type.BINARY)
        continuous_indices = self._get_variable_indices(problem, Variable.Type.CONTINUOUS)

        # 2. binary and continuous variables are separable in objective
        for bin_int_index in bin_int_indices:
            for continuous_index in continuous_indices:
                coeff = problem.objective.quadratic[bin_int_index, continuous_index]
                if coeff != 0:
                    # binary and continuous vars are mixed.
                    msg += 'Binary and continuous variables are not separable in the objective. '

        # if an error occurred, return error message, otherwise, return None
        return msg

    def solve(self, problem: QuadraticProgram) -> ADMMOptimizationResult:
        """Tries to solves the given problem using ADMM algorithm.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is not compatible with the ADMM optimizer.
        """
        self._verify_compatibility(problem)

        # debug
        self._log.debug("Initial problem: %s", problem.export_as_lp_string())

        # map integer variables to binary variables
        from ..converters.integer_to_binary import IntegerToBinary
        int2bin = IntegerToBinary()
        original_problem = problem
        problem = int2bin.convert(problem)

        # we deal with minimization in the optimizer, so turn the problem to minimization
        problem, sense = self._turn_to_minimization(problem)

        # create our computation state.
        self._state = ADMMState(problem, self._params.rho_initial)

        # parse problem and convert to an ADMM specific representation.
        self._state.binary_indices = self._get_variable_indices(problem, Variable.Type.BINARY)
        self._state.continuous_indices = self._get_variable_indices(problem,
                                                                    Variable.Type.CONTINUOUS)
        if self._params.warm_start:
            # warm start injection for the initial values of the variables
            self._warm_start(problem)

        # convert optimization problem to a set of matrices and vector that are used
        # at each iteration.
        self._convert_problem_representation()

        start_time = time.time()
        # we have not stated our computations yet, so elapsed time initialized as zero.
        elapsed_time = 0.0
        iteration = 0
        residual = 1.e+2

        while (iteration < self._params.maxiter and residual > self._params.tol) \
                and (elapsed_time < self._params.max_time):
            if self._state.step1_absolute_indices:
                op1 = self._create_step1_problem()
                self._state.x0 = self._update_x0(op1)
                # debug
                self._log.debug("Step 1 sub-problem: %s", op1.export_as_lp_string())
            # else, no binary variables exist, and no update to be done in this case.
            # debug
            self._log.debug("x0=%s", self._state.x0)

            op2 = self._create_step2_problem()
            self._state.u, self._state.z = self._update_x1(op2)
            # debug
            self._log.debug("Step 2 sub-problem: %s", op2.export_as_lp_string())
            self._log.debug("u=%s", self._state.u)
            self._log.debug("z=%s", self._state.z)

            if self._params.three_block:
                if self._state.binary_indices:
                    op3 = self._create_step3_problem()
                    self._state.y = self._update_y(op3)
                    # debug
                    self._log.debug("Step 3 sub-problem: %s", op3.export_as_lp_string())
                # debug
                self._log.debug("y=%s", self._state.y)

            self._state.lambda_mult = self._update_lambda_mult()
            # debug
            self._log.debug("lambda: %s", self._state.lambda_mult)

            cost_iterate = self._get_objective_value()
            constraint_residual = self._get_constraint_residual()
            residual, dual_residual = self._get_solution_residuals(iteration)
            merit = self._get_merit(cost_iterate, constraint_residual)
            # debug
            self._log.debug("cost_iterate=%s, cr=%s, merit=%s",
                            cost_iterate, constraint_residual, merit)

            # costs are saved with their original sign.
            self._state.cost_iterates.append(cost_iterate)
            self._state.residuals.append(residual)
            self._state.dual_residuals.append(dual_residual)
            self._state.cons_r.append(constraint_residual)
            self._state.merits.append(merit)
            self._state.lambdas.append(np.linalg.norm(self._state.lambda_mult))

            self._state.x0_saved.append(self._state.x0)
            self._state.u_saved.append(self._state.u)
            self._state.z_saved.append(self._state.z)
            self._state.z_saved.append(self._state.y)

            self._update_rho(residual, dual_residual)

            iteration += 1
            elapsed_time = time.time() - start_time

        binary_vars, continuous_vars, objective_value = self._get_best_merit_solution()
        solution = self._revert_solution_indexes(binary_vars, continuous_vars)

        # flip the objective sign again if required
        objective_value = objective_value * sense

        self._log.debug("solution=%s, objective=%s at iteration=%s",
                        solution, objective_value, iteration)

        # convert back integer to binary
        # `state` is our internal state of computations.
        return cast(ADMMOptimizationResult,
                    self._interpret(x=solution, converters=int2bin, problem=original_problem,
                                    result_class=ADMMOptimizationResult,
                                    state=self._state))

    @staticmethod
    def _turn_to_minimization(problem: QuadraticProgram) -> Tuple[QuadraticProgram, float]:
        """
        Turns the problem to `ObjSense.MINIMIZE` by flipping the sign of the objective function
        if initially it is `ObjSense.MAXIMIZE`. Otherwise returns the original problem.

        Args:
            problem: a problem to turn to minimization.

        Returns:
            A copy of the problem if sign flip is required, otherwise the original problem and
            the original sense of the problem in the numerical representation.
        """
        sense = problem.objective.sense.value
        if problem.objective.sense == QuadraticObjective.Sense.MAXIMIZE:
            problem = copy.deepcopy(problem)
            problem.objective.sense = QuadraticObjective.Sense.MINIMIZE
            problem.objective.constant = (-1) * problem.objective.constant
            problem.objective.linear = (-1) * problem.objective.linear.coefficients
            problem.objective.quadratic = (-1) * problem.objective.quadratic.coefficients
        return problem, sense

    @staticmethod
    def _get_variable_indices(op: QuadraticProgram, var_type: VarType) -> List[int]:
        """Returns a list of indices of the variables of the specified type.

        Args:
            op: Optimization problem.
            var_type: type of variables to look for.

        Returns:
            List of indices.
        """
        indices = []
        for i, variable in enumerate(op.variables):
            if variable.vartype == var_type:
                indices.append(i)

        return indices

    def _get_current_solution(self) -> np.ndarray:
        """
        Returns current solution of the problem.

        Returns:
            An array of the current solution.
        """
        return self._revert_solution_indexes(self._state.x0, self._state.u)

    def _revert_solution_indexes(self, binary_vars: np.ndarray, continuous_vars: np.ndarray) \
            -> np.ndarray:
        """Constructs a solution array where variables are stored in the correct order.

        Args:
            binary_vars: solution for binary variables
            continuous_vars: solution for continuous variables

        Returns:
            A solution array.
        """
        solution = np.zeros(len(self._state.binary_indices) + len(self._state.continuous_indices))
        # restore solution at the original index location
        solution.put(self._state.binary_indices, binary_vars)
        solution.put(self._state.continuous_indices, continuous_vars)
        return solution

    def _convert_problem_representation(self) -> None:
        """Converts problem representation into set of matrices and vectors."""
        binary_var_indices = set(self._state.binary_indices)
        # separate constraints
        for l_constraint in self._state.op.linear_constraints:
            if l_constraint.sense == Constraint.Sense.EQ:
                self._state.equality_constraints.append(l_constraint)

                # verify that there are only binary variables in the constraint
                # this is to build A0, b0 in step 1
                constraint_var_indices = set(l_constraint.linear.to_dict().keys())
                if constraint_var_indices.issubset(binary_var_indices):
                    self._state.binary_equality_constraints.append(l_constraint)

            elif l_constraint.sense in (Constraint.Sense.LE, Constraint.Sense.GE):
                self._state.inequality_constraints.append(l_constraint)

        # separate quadratic constraints into eq and non-eq
        for q_constraint in self._state.op.quadratic_constraints:
            if q_constraint.sense == Constraint.Sense.EQ:
                self._state.equality_constraints.append(q_constraint)
            elif q_constraint.sense in (Constraint.Sense.LE, Constraint.Sense.GE):
                self._state.inequality_constraints.append(q_constraint)

        # separately keep binary variables that are for step 1 only
        # temp variables are due to limit of 100 chars per line
        step1_absolute_indices, step1_relative_indices = self._get_step1_indices()
        self._state.step1_absolute_indices = step1_absolute_indices
        self._state.step1_relative_indices = step1_relative_indices

        # objective
        self._state.q0 = self._get_q(self._state.step1_absolute_indices)
        c0_vec = self._state.op.objective.linear.to_array()[self._state.step1_absolute_indices]
        self._state.c0 = c0_vec
        self._state.q1 = self._get_q(self._state.continuous_indices)
        self._state.c1 = self._state.op.objective.linear.to_array()[self._state.continuous_indices]
        # equality constraints with binary vars only
        self._state.a0, self._state.b0 = self._get_a0_b0()

    def _get_step1_indices(self) -> Tuple[List[int], List[int]]:
        """
        Constructs two arrays of absolute (pointing to the original problem) and relative (pointing
        to the list of all binary variables) indices of the variables considered
        to be included in the step1(QUBO) problem.

        Returns: A tuple of lists with absolute and relative indices
        """
        # here we keep binary indices from the original problem
        step1_absolute_indices = []

        # iterate over binary variables and put all binary variables mentioned in the objective
        # to the array for the step1
        for binary_index in self._state.binary_indices:
            # here we check if this binary variable present in the objective
            # either in the linear or quadratic terms
            if self._state.op.objective.linear[binary_index] != 0 or np.abs(
                    self._state.op.objective.quadratic.coefficients[binary_index, :]).sum() != 0:
                # add the variable if it was not added before
                if binary_index not in step1_absolute_indices:
                    step1_absolute_indices.append(binary_index)

        # compute all unverified binary variables (the variables that are present in constraints
        # but not in objective):
        #   rest variables := all binary variables - already verified for step 1
        rest_binary = set(self._state.binary_indices).difference(step1_absolute_indices)

        # verify if an equality contains binary variables
        for constraint in self._state.binary_equality_constraints:
            for binary_index in list(rest_binary):
                if constraint.linear[binary_index] > 0 \
                        and binary_index not in step1_absolute_indices:
                    # a binary variable with the binary_index is present in this constraint
                    step1_absolute_indices.append(binary_index)

        # we want to preserve order of the variables but this order could be broken by adding
        # a variable in the previous for loop.
        step1_absolute_indices.sort()

        # compute relative indices, these indices are used when we generate step1 and
        # update variables on step1.
        # on step1 we solve for a subset of all binary variables,
        # so we want to operate only these indices
        step1_relative_indices = []
        relative_index = 0
        # for each binary variable that comes from lin.eq/obj and which is denoted by abs_index
        for abs_index in step1_absolute_indices:
            found = False
            # we want to find relative index of a variable the comes from linear constraints
            # or objective across all binary variables
            for j in range(relative_index, len(self._state.binary_indices)):
                if self._state.binary_indices[j] == abs_index:
                    found = True
                    relative_index = j
                    break
            if found:
                step1_relative_indices.append(relative_index)
            else:
                raise ValueError("No relative index found!")

        return step1_absolute_indices, step1_relative_indices

    def _get_q(self, variable_indices: List[int]) -> np.ndarray:
        """Constructs a quadratic matrix for the variables with the specified indices
        from the quadratic terms in the objective.

        Args:
            variable_indices: variable indices to look for.

        Returns:
            A matrix as a numpy array of the shape(len(variable_indices), len(variable_indices)).
        """
        size = len(variable_indices)
        q = np.zeros(shape=(size, size))
        # fill in the matrix
        # in fact we use re-indexed variables
        # we build upper triangular matrix to avoid doubling of the coefficients
        for i in range(0, size):
            for j in range(i, size):
                q[i, j] = \
                    self._state.op.objective.quadratic[variable_indices[i], variable_indices[j]]

        return q

    def _get_a0_b0(self) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs a matrix and a vector from the constraints in a form of Ax = b, where
        x is a vector of binary variables.

        Returns:
            Corresponding matrix and vector as numpy arrays.

        Raises:
            ValueError: if the problem is not suitable for this optimizer.
        """
        matrix = []
        vector = []

        for constraint in self._state.binary_equality_constraints:
            row = constraint.linear.to_array().take(self._state.step1_absolute_indices).tolist()

            matrix.append(row)
            vector.append(constraint.rhs)

        if len(matrix) != 0:
            np_matrix = np.array(matrix)
            np_vector = np.array(vector)
        else:
            np_matrix = np.array([0] * len(self._state.step1_absolute_indices)).reshape((1, -1))
            np_vector = np.zeros(shape=(1,))

        return np_matrix, np_vector

    def _create_step1_problem(self) -> QuadraticProgram:
        """Creates a step 1 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op1 = QuadraticProgram()

        binary_size = len(self._state.step1_absolute_indices)
        # create the same binary variables.
        for i in range(binary_size):
            name = self._state.op.variables[self._state.step1_absolute_indices[i]].name
            op1.binary_var(name=name)

        # prepare and set quadratic objective.
        quadratic_objective = self._state.q0 + \
            self._params.factor_c / 2 * np.dot(self._state.a0.transpose(), self._state.a0) + \
            self._state.rho / 2 * np.eye(binary_size)
        op1.objective.quadratic = quadratic_objective

        # prepare and set linear objective.
        linear_objective = self._state.c0 - \
            self._params.factor_c * np.dot(self._state.b0, self._state.a0) + \
            self._state.rho * (- self._state.y[self._state.step1_relative_indices] -
                               self._state.z[self._state.step1_relative_indices]) + \
            self._state.lambda_mult[self._state.step1_relative_indices]

        op1.objective.linear = linear_objective
        return op1

    def _create_step2_problem(self) -> QuadraticProgram:
        """Creates a step 2 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op2 = copy.deepcopy(self._state.op)
        # replace binary variables with the continuous ones bound in [0,1]
        # x0(bin) -> z(cts)
        # u (cts) are still there unchanged
        for i, var_index in enumerate(self._state.binary_indices):
            variable = op2.variables[var_index]
            variable.vartype = Variable.Type.CONTINUOUS
            variable.upperbound = 1.
            variable.lowerbound = 0.
            # replacing Q0 objective and take of min/max sense, initially we consider minimization
            op2.objective.quadratic[var_index, var_index] = self._state.rho / 2
            # replacing linear objective
            op2.objective.linear[var_index] = -1 * self._state.lambda_mult[i] - self._state.rho * \
                (self._state.x0[i] - self._state.y[i])

        # remove A0 x0 = b0 constraints
        for constraint in self._state.binary_equality_constraints:
            op2.remove_linear_constraint(constraint.name)

        return op2

    def _create_step3_problem(self) -> QuadraticProgram:
        """Creates a step 3 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op3 = QuadraticProgram()
        # add y variables.
        binary_size = len(self._state.binary_indices)
        for i in range(binary_size):
            name = self._state.op.variables[self._state.binary_indices[i]].name
            op3.continuous_var(lowerbound=-np.inf, upperbound=np.inf, name=name)

        # set quadratic objective y
        quadratic_y = self._params.beta / 2 * np.eye(binary_size) + \
            self._state.rho / 2 * np.eye(binary_size)
        op3.objective.quadratic = quadratic_y

        # set linear objective for y
        linear_y = - self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.z)
        op3.objective.linear = linear_y

        return op3

    def _update_x0(self, op1: QuadraticProgram) -> np.ndarray:
        """Solves the Step1 QuadraticProgram via the qubo optimizer.

        Args:
            op1: the Step1 QuadraticProgram.

        Returns:
            A solution of the Step1, as a numpy array.
        """
        x0_all_binaries = np.zeros(len(self._state.binary_indices))
        x0_qubo = np.asarray(self._qubo_optimizer.solve(op1).x)
        x0_all_binaries[self._state.step1_relative_indices] = x0_qubo
        return x0_all_binaries

    def _update_x1(self, op2: QuadraticProgram) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the Step2 QuadraticProgram via the continuous optimizer.

        Args:
            op2: the Step2 QuadraticProgram

        Returns:
            A solution of the Step2, as a pair of numpy arrays.
            First array contains the values of decision variables u, and
            second array contains the values of decision variables z.

        """
        vars_op2 = np.asarray(self._continuous_optimizer.solve(op2).x)
        vars_u = vars_op2.take(self._state.continuous_indices)
        vars_z = vars_op2.take(self._state.binary_indices)
        return vars_u, vars_z

    def _update_y(self, op3: QuadraticProgram) -> np.ndarray:
        """Solves the Step3 QuadraticProgram via the continuous optimizer.

        Args:
            op3: the Step3 QuadraticProgram

        Returns:
            A solution of the Step3, as a numpy array.

        """
        return np.asarray(self._continuous_optimizer.solve(op3).x)

    def _get_best_merit_solution(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """The ADMM solution is that for which the merit value is the min
            * sol: Iterate with the min merit value
            * sol_val: Value of sol, according to the original objective

        Returns:
            A tuple of (binary_vars, continuous_vars, sol_val), where
                * binary_vars: binary variable values with the min merit value
                * continuous_vars: continuous variable values with the min merit value
                * sol_val: Value of the objective function
        """

        it_min_merits = self._state.merits.index(min(self._state.merits))
        binary_vars = self._state.x0_saved[it_min_merits]
        continuous_vars = self._state.u_saved[it_min_merits]
        sol_val = self._state.cost_iterates[it_min_merits]
        return binary_vars, continuous_vars, sol_val

    def _update_lambda_mult(self) -> np.ndarray:
        """
        Updates the values of lambda multiplier, given the updated iterates
        x0, z, and y.

        Returns: The updated array of values of lambda multiplier.

        """
        return self._state.lambda_mult + \
            self._state.rho * (self._state.x0 - self._state.z - self._state.y)

    def _update_rho(self, primal_residual: float, dual_residual: float) -> None:
        """Updating the rho parameter in ADMM.

        Args:
            primal_residual: primal residual
            dual_residual: dual residual
        """

        if self._params.vary_rho == UPDATE_RHO_BY_TEN_PERCENT:
            # Increase rho, to aid convergence.
            if self._state.rho < 1.e+10:
                self._state.rho *= 1.1
        elif self._params.vary_rho == UPDATE_RHO_BY_RESIDUALS:
            if primal_residual > self._params.mu_res * dual_residual:
                self._state.rho = self._params.tau_incr * self._state.rho
            elif dual_residual > self._params.mu_res * primal_residual:
                self._state.rho = self._params.tau_decr * self._state.rho

    def _get_constraint_residual(self) -> float:
        """Compute violation of the constraints of the original problem, as:
            * norm 1 of the body-rhs of eq. constraints
            * -1 * min(body - rhs, 0) for geq constraints
            * max(body - rhs, 0) for leq constraints

        Returns:
            Violation of the constraints as a float value
        """
        solution = self._get_current_solution()
        # equality constraints
        cr_eq = 0
        for constraint in self._state.equality_constraints:
            cr_eq += np.abs(constraint.evaluate(solution) - constraint.rhs)

        # inequality constraints
        cr_ineq = 0.0
        for constraint in self._state.inequality_constraints:
            sense = -1.0 if constraint.sense == Constraint.Sense.GE else 1.0
            cr_ineq += max(sense * (constraint.evaluate(solution) - constraint.rhs), 0.0)

        return cr_eq + cr_ineq

    def _get_merit(self, cost_iterate: float, constraint_residual: float) -> float:
        """Compute merit value associated with the current iterate

        Args:
            cost_iterate: Cost at the certain iteration.
            constraint_residual: Value of violation of the constraints.

        Returns:
            Merit value as a float
        """
        return cost_iterate + self._params.mu_merit * constraint_residual

    def _get_objective_value(self) -> float:
        """Computes the value of the objective function.

        Returns:
            Value of the objective function as a float
        """
        return self._state.op.objective.evaluate(self._get_current_solution())

    def _get_solution_residuals(self, iteration: int) -> Tuple[float, float]:
        """Compute primal and dual residual.

        Args:
            iteration: Iteration number.

        Returns:
            r, s as primary and dual residuals.
        """
        elements = self._state.x0 - self._state.z - self._state.y
        primal_residual = np.linalg.norm(elements)
        if iteration > 0:
            elements_dual = self._state.z - self._state.z_saved[iteration - 1]
        else:
            elements_dual = self._state.z - self._state.z_init
        dual_residual = self._state.rho * np.linalg.norm(elements_dual)

        return primal_residual, dual_residual

    def _warm_start(self, problem: QuadraticProgram) -> None:
        """Solves a relaxed (all variables are continuous) and initializes the optimizer state with
            the found solution.

        Args:
            problem: a problem to solve.

        Returns:
            None
        """
        qp_copy = copy.deepcopy(problem)
        for variable in qp_copy.variables:
            variable.vartype = VarType.CONTINUOUS
        cts_result = self._continuous_optimizer.solve(qp_copy)
        logger.debug("Continuous relaxation: %s", cts_result.x)

        self._state.x0 = cts_result.x[self._state.binary_indices]
        self._state.u = cts_result.x[self._state.continuous_indices]
        self._state.z = cts_result.x[self._state.binary_indices]

    @property
    def parameters(self) -> ADMMParameters:
        """Returns current parameters of the optimizer.

        Returns:
            The parameters.
        """
        return self._params

    @parameters.setter
    def parameters(self, params: ADMMParameters) -> None:
        """Sets the parameters of the optimizer.

        Args:
            params: New parameters to set.
        """
        self._params = params

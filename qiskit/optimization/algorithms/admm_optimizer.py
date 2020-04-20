# -*- coding: utf-8 -*-

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
import logging
import time
from typing import List, Optional, Any
import numpy as np
from scipy.linalg import block_diag

from qiskit.optimization.algorithms.cplex_optimizer import CplexOptimizer
from qiskit.optimization.algorithms.optimization_algorithm import (OptimizationAlgorithm,
                                                                   OptimizationResult)
from qiskit.optimization.problems import VarType, ConstraintSense
from qiskit.optimization.problems.quadratic_program import QuadraticProgram

UPDATE_RHO_BY_TEN_PERCENT = 0
UPDATE_RHO_BY_RESIDUALS = 1

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import SparsePair, SparseTriple
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


class ADMMParameters:
    """Defines a set of parameters for ADMM optimizer."""

    def __init__(self, rho_initial: float = 10000, factor_c: float = 100000, beta: float = 1000,
                 max_iter: int = 10, tol: float = 1.e-4, max_time: float = np.inf,
                 three_block: bool = True, vary_rho: int = UPDATE_RHO_BY_TEN_PERCENT,
                 tau_incr: float = 2, tau_decr: float = 2, mu_res: float = 10,
                 mu_merit: float = 1000) -> None:
        """Defines parameters for ADMM optimizer and their default values.

        Args:
            rho_initial: Initial value of rho parameter of ADMM.
            factor_c: Penalizing factor for equality constraints, when mapping to QUBO.
            beta: Penalization for y decision variables.
            max_iter: Maximum number of iterations for ADMM.
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
        """
        super().__init__()
        self.mu_merit = mu_merit
        self.mu_res = mu_res
        self.tau_decr = tau_decr
        self.tau_incr = tau_incr
        self.vary_rho = vary_rho
        self.three_block = three_block
        self.max_time = max_time
        self.tol = tol
        self.max_iter = max_iter
        self.factor_c = factor_c
        self.beta = beta
        self.rho_initial = rho_initial


class ADMMState:
    """Internal computation state of the ADMM implementation.

    The state keeps track of various variables are stored that are being updated during problem
    solving. The values are relevant to the problem being solved. The state is recreated for each
    optimization problem. State is returned as the third value.
    """

    def __init__(self,
                 op: QuadraticProgram,
                 binary_indices: List[int],
                 continuous_indices: List[int],
                 rho_initial: float) -> None:
        """Constructs an internal computation state of the ADMM implementation.

        Args:
            op: The optimization problem being solved.
            binary_indices: Indices of the binary decision variables of the original problem.
            continuous_indices: Indices of the continuous decision variables of the original
             problem.
            rho_initial: Initial value of the rho parameter.
        """
        super().__init__()

        # Optimization problem itself
        self.op = op
        # Indices of the variables
        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices
        self.sense = op.objective.sense.value

        # define heavily used matrix, they are used at each iteration, so let's cache them,
        # they are np.ndarrays
        # pylint:disable=invalid-name
        # objective
        self.q0 = None
        self.c0 = None
        self.q1 = None
        self.c1 = None
        # constraints
        self.a0 = None
        self.b0 = None
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.a3 = None
        self.b2 = None
        self.a4 = None
        self.b3 = None

        # These are the parameters that are updated in the ADMM iterations.
        self.u: np.ndarray = np.zeros(len(continuous_indices))
        binary_size = len(binary_indices)
        self.x0: np.ndarray = np.zeros(binary_size)
        self.z: np.ndarray = np.zeros(binary_size)
        self.z_init: np.ndarray = self.z
        self.y: np.ndarray = np.zeros(binary_size)
        self.lambda_mult: np.ndarray = np.zeros(binary_size)

        # The following structures store quantities obtained in each ADMM iteration.
        self.cost_iterates = []
        self.residuals = []
        self.dual_residuals = []
        self.cons_r = []
        self.merits = []
        self.lambdas = []
        self.x0_saved = []
        self.u_saved = []
        self.z_saved = []
        self.y_saved = []
        self.rho = rho_initial


class ADMMOptimizerResult(OptimizationResult):
    """ ADMMOptimizer Result."""

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 state: Optional[ADMMState] = None, results: Optional[Any] = None) -> None:
        super().__init__(x, fval, results or state)
        self._state = state

    @property
    def state(self) -> Optional[ADMMState]:
        """ returns state """
        return self._state


class ADMMOptimizer(OptimizationAlgorithm):
    
    """An implementation of the ADMM-based heuristic introduced here:
    Gambella, C., & Simonetto, A. (2020).
     Multi-block ADMM Heuristics for Mixed-Binary Optimization on Classical and Quantum Computers.
     arXiv preprint arXiv:2001.02069.
    """

    def __init__(self, qubo_optimizer: Optional[OptimizationAlgorithm] = None,
                 continuous_optimizer: Optional[OptimizationAlgorithm] = None,
                 params: Optional[ADMMParameters] = None) -> None:
        """Constructs an instance of ADMMOptimizer.

        Args:
            qubo_optimizer: An instance of OptimizationAlgorithm that can effectively solve
                QUBO problems.
            continuous_optimizer: An instance of OptimizationAlgorithm that can solve
                continuous problems.
            params: An instance of ADMMParameters.

        Raises:
            NameError: CPLEX is not installed.
        """
        if not _HAS_CPLEX:
            raise NameError('CPLEX is not installed.')

        super().__init__()
        self._log = logging.getLogger(__name__)

        # create default params if not present
        self._params = params or ADMMParameters()

        # create optimizers if not specified
        self._qubo_optimizer = qubo_optimizer or CplexOptimizer()
        self._continuous_optimizer = continuous_optimizer or CplexOptimizer()

        # internal state where we'll keep intermediate solution
        # here, we just declare the class variable, the variable is initialized in kept in
        # the solve method.
        self._state: Optional[ADMMState] = None

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

        # 1. only binary and continuous variables are supported
        for variable in problem.variables:
            if variable.vartype not in (VarType.binary, VarType.continuous):
                # variable is not binary and not continuous.
                msg += 'Only binary and continuous variables are supported. '

        binary_indices = self._get_variable_indices(problem, VarType.binary)
        continuous_indices = self._get_variable_indices(problem, VarType.continuous)

        # 2. binary and continuous variables are separable in objective
        for binary_index in binary_indices:
            for continuous_index in continuous_indices:
                coeff = problem.objective.get_quadratic_coefficients(binary_index, continuous_index)
                if coeff != 0:
                    # binary and continuous vars are mixed.
                    msg += 'Binary and continuous variables are not separable in the objective. '

        # 3. no quadratic constraints are supported.
        quad_constraints = len(problem.quadratic_constraints)
        if quad_constraints is not None and quad_constraints > 0:
            # quadratic constraints are not supported.
            msg += 'Quadratic constraints are not supported. '

        # if an error occurred, return error message, otherwise, return None
        return msg

    def solve(self, problem: QuadraticProgram) -> ADMMOptimizerResult:
        """Tries to solves the given problem using ADMM algorithm.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        # parse problem and convert to an ADMM specific representation.
        binary_indices = self._get_variable_indices(problem, VarType.BINARY)
        continuous_indices = self._get_variable_indices(problem, VarType.CONTINUOUS)

        # create our computation state.
        self._state = ADMMState(problem, binary_indices,
                                continuous_indices, self._params.rho_initial)

        # convert optimization problem to a set of matrices and vector that are used
        # at each iteration.
        self._convert_problem_representation()

        start_time = time.time()
        # we have not stated our computations yet, so elapsed time initialized as zero.
        elapsed_time = 0
        iteration = 0
        residual = 1.e+2

        while (iteration < self._params.max_iter and residual > self._params.tol) \
                and (elapsed_time < self._params.max_time):
            if binary_indices:
                op1 = self._create_step1_problem()

                self._state.x0 = self._update_x0(op1)
            # else, no binary variables exist,
            # and no update to be done in this case.
            # debug
            self._log.debug("x0=%s", self._state.x0)

            op2 = self._create_step2_problem()
            self._state.u, self._state.z = self._update_x1(op2)
            # debug
            self._log.debug("u=%s", self._state.u)
            self._log.debug("z=%s", self._state.z)

            if self._params.three_block:
                if binary_indices:
                    op3 = self._create_step3_problem()
                    self._state.y = self._update_y(op3)
                # debug
                self._log.debug("y=%s", self._state.y)

            self._state.lambda_mult = self._update_lambda_mult()

            cost_iterate = self._get_objective_value()
            constraint_residual = self._get_constraint_residual()
            residual, dual_residual = self._get_solution_residuals(iteration)
            merit = self._get_merit(cost_iterate, constraint_residual)
            # debug
            self._log.debug("cost_iterate=%s, cr=%s, merit=%s",
                            cost_iterate, constraint_residual, merit)

            # costs and merits are saved with their original sign.
            self._state.cost_iterates.append(self._state.sense * cost_iterate)
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

        solution, objective_value = self._get_best_merit_solution()
        solution = self._revert_solution_indexes(solution)

        # third parameter is our internal state of computations.
        result = ADMMOptimizerResult(solution, objective_value, self._state)
        # debug
        self._log.debug("solution=%s, objective=%s at iteration=%s",
                        solution, objective_value, iteration)
        return result

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

    def _revert_solution_indexes(self, internal_solution: List[np.ndarray]) \
            -> np.ndarray:
        """Constructs a solution array where variables are stored in the correct order.

        Args:
            internal_solution: a list with two lists: solutions for binary variables and
                for continuous variables.

        Returns:
            A solution array.
        """
        binary_solutions, continuous_solutions = internal_solution
        solution = np.zeros(len(self._state.binary_indices) + len(self._state.continuous_indices))
        # restore solution at the original index location
        for i, binary_index in enumerate(self._state.binary_indices):
            solution[binary_index] = binary_solutions[i]
        for i, continuous_index in enumerate(self._state.continuous_indices):
            solution[continuous_index] = continuous_solutions[i]
        return solution

    def _convert_problem_representation(self) -> None:
        """Converts problem representation into set of matrices and vectors.
        Specifically, the optimization problem is represented as:

        min_{x0, u} x0^T q0 x0 + c0^T x0 + u^T q1 u + c1^T u

        s.t. a0 x0 = b0
            a1 x0 <= b1
            a2 z + a3 u <= b2
            a4 u <= b3

        """
        # objective
        self._state.q0 = self._get_q(self._state.binary_indices)
        self._state.c0 = self._get_c(self._state.binary_indices)
        self._state.q1 = self._get_q(self._state.continuous_indices)
        self._state.c1 = self._get_c(self._state.continuous_indices)
        # constraints
        self._state.a0, self._state.b0 = self._get_a0_b0()
        self._state.a1, self._state.b1 = self._get_a1_b1()
        self._state.a2, self._state.a3, self._state.b2 = self._get_a2_a3_b2()
        self._state.a4, self._state.b3 = self._get_a4_b3()

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
        for i, var_index_i in enumerate(variable_indices):
            for j, var_index_j in enumerate(variable_indices):
                # coefficients_as_array
                q[i, j] = self._state.op.objective.quadratic[var_index_i, var_index_j]

        # flip the sign, according to the optimization sense, e.g. sense == 1 if minimize,
        # sense == -1 if maximize.
        return q * self._state.sense

    def _get_c(self, variable_indices: List[int]) -> np.ndarray:
        """Constructs a vector for the variables with the specified indices from the linear terms
        in the objective.

        Args:
            variable_indices: variable indices to look for.

        Returns:
            A numpy array of the shape(len(variable_indices)).
        """
        # c = np.array(self._state.op.objective.get_linear(variable_indices))
        c = self._state.op.objective.linear.to_array().take(variable_indices)
        # flip the sign, according to the optimization sense, e.g. sense == 1 if minimize,
        # sense == -1 if maximize.
        c *= self._state.sense
        return c

    def _assign_row_values(self, matrix: List[List[float]], vector: List[float],
                           constraint_index: int, variable_indices: List[int]):
        """Appends a row to the specified matrix and vector based on the constraint specified by
        the index using specified variables.

        Args:
            matrix: a matrix to extend.
            vector: a vector to expand.
            constraint_index: constraint index to look for.
            variable_indices: variables to look for.

        Returns:
            None
        """
        # assign matrix row.
        row = []
        # todo: replace with .take()
        for var_index in variable_indices:
            # row.append(self._state.op
            #            .linear_constraints.get_coefficients(constraint_index, var_index))
            row.append(
                self._state.op.linear_constraints[constraint_index].linear[var_index])
        matrix.append(row)

        # assign vector row.
        vector.append(self._state.op.linear_constraints[constraint_index].rhs)

        # flip the sign if constraint is G, we want L constraints.
        if self._state.op.linear_constraints[constraint_index].sense == ConstraintSense.GE:
            # invert the sign to make constraint "L".
            matrix[-1] = [-1 * el for el in matrix[-1]]
            vector[-1] = -1 * vector[-1]

    @staticmethod
    def _create_ndarrays(matrix: List[List[float]], vector: List[float], size: int) \
            -> (np.ndarray, np.ndarray):
        """Converts representation of a matrix and a vector in form of lists to numpy array.

        Args:
            matrix: matrix to convert.
            vector: vector to convert.
            size: size to create matrix and vector.

        Returns:
            Converted matrix and vector as numpy arrays.
        """
        # if we don't have such constraints, return just dummy arrays.
        if len(matrix) != 0:
            return np.array(matrix), np.array(vector)
        else:
            return np.array([0] * size).reshape((1, -1)), np.zeros(shape=(1,))

    def _get_a0_b0(self) -> (np.ndarray, np.ndarray):
        """Constructs a matrix and a vector from the constraints in a form of Ax = b, where
        x is a vector of binary variables.

        Returns:
            Corresponding matrix and vector as numpy arrays.

        Raises:
            ValueError: if the problem is not suitable for this optimizer.
        """
        matrix = []
        vector = []

        index_set = set(self._state.binary_indices)
        for constraint_index, constraint in enumerate(self._state.op.linear_constraints):
            # we check only equality constraints here.
            if constraint.sense != ConstraintSense.EQ:
                continue
            # todo: add condition that only binary variables are in the constraint
            # row = self._state.op.linear_constraints.get_rows(constraint_index)
            row = self._state.op.linear_constraints[constraint_index].linear.to_array().take(self._state.binary_indices)
            self._assign_row_values(matrix, vector,
                                    constraint_index, self._state.binary_indices)
            # if set(row.ind).issubset(index_set):
            #     self._assign_row_values(matrix, vector,
            #                             constraint_index, self._state.binary_indices)
            # else:
            #     raise ValueError(
            #         "Linear constraint with the 'E' sense must contain only binary variables, "
            #         "row indices: {}, binary variable indices: {}".format(
            #             row, self._state.binary_indices))

        return self._create_ndarrays(matrix, vector, len(self._state.binary_indices))

    def _get_inequality_matrix_and_vector(self, variable_indices: List[int]) \
            -> (List[List[float]], List[float]):
        """Constructs a matrix and a vector from the constraints in a form of Ax <= b, where
        x is a vector of variables specified by the indices.

        Args:
            variable_indices: variable indices to look for.

        Returns:
            A list based representation of the matrix and the vector.
        """
        matrix = []
        vector = []

        index_set = set(variable_indices)
        for constraint_index, constraint in enumerate(self._state.op.linear_constraints):
            # todo: missing "R"
            if constraint.sense in [ConstraintSense.EQ]:
                # TODO: Ranged constraints should be supported
                continue
            constraint_indices = set(self._state.op.linear_constraints[constraint_index].linear.to_dict().keys())
            # row = self._state.op.linear_constraints.get_rows(constraint_index)
            if constraint_indices.issubset(index_set):
                self._assign_row_values(matrix, vector, constraint_index, variable_indices)

        return matrix, vector

    def _get_a1_b1(self) -> (np.ndarray, np.ndarray):
        """Constructs a matrix and a vector from the constraints in a form of Ax <= b, where
        x is a vector of binary variables.

        Returns:
            A numpy based representation of the matrix and the vector.
        """
        matrix, vector = self._get_inequality_matrix_and_vector(self._state.binary_indices)
        return self._create_ndarrays(matrix, vector, len(self._state.binary_indices))

    def _get_a4_b3(self) -> (np.ndarray, np.ndarray):
        """Constructs a matrix and a vector from the constraints in a form of Au <= b, where
        u is a vector of continuous variables.

        Returns:
            A numpy based representation of the matrix and the vector.
        """
        matrix, vector = self._get_inequality_matrix_and_vector(self._state.continuous_indices)
        return self._create_ndarrays(matrix, vector, len(self._state.continuous_indices))

    def _get_a2_a3_b2(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Constructs matrices and a vector from the constraints in a form of A_2x + A_3u <= b,
        where x is a vector of binary variables and u is a vector of continuous variables.

        Returns:
            A numpy representation of two matrices and one vector.
        """
        matrix = []
        vector = []

        binary_index_set = set(self._state.binary_indices)
        continuous_index_set = set(self._state.continuous_indices)
        all_variables = self._state.binary_indices + self._state.continuous_indices
        for constraint_index, constraint in enumerate(self._state.op.linear_constraints):
            # todo: "R" constraints
            if constraint.sense in [ConstraintSense.EQ]:
                # TODO: Ranged constraints should be supported as well
                continue
            # sense either G or L.
            # row = self._state.op.linear_constraints.get_rows(constraint_index)
            constraint_indices = set(
                self._state.op.linear_constraints[constraint_index].linear.to_dict().keys())
            # row_indices = set(row.ind)
            # we must have a least one binary and one continuous variable,
            # otherwise it is another type of constraints.
            if len(constraint_indices & binary_index_set) != 0 and len(
                    constraint_indices & continuous_index_set) != 0:
                self._assign_row_values(matrix, vector, constraint_index, all_variables)

        matrix, b_2 = self._create_ndarrays(matrix, vector, len(all_variables))
        # a2
        a_2 = matrix[:, 0:len(self._state.binary_indices)]
        a_3 = matrix[:, len(self._state.binary_indices):]
        return a_2, a_3, b_2

    def _create_step1_problem(self) -> QuadraticProgram:
        """Creates a step 1 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op1 = QuadraticProgram()

        binary_size = len(self._state.binary_indices)
        # create the same binary variables.
        for i in range(binary_size):
            op1.binary_var(name="x0_" + str(i + 1))
        # op1.variables.add(names=["x0_" + str(i + 1) for i in range(binary_size)],
        #                   types=["I"] * binary_size,
        #                   lb=[0.] * binary_size,
        #                   ub=[1.] * binary_size)

        # prepare and set quadratic objective.
        # NOTE: The multiplication by 2 is needed for the solvers to parse
        # the quadratic coefficients.
        # todo: do we need multiplication by two? For anything other than self._state.q0
        quadratic_objective = self._state.q0 +\
                self._params.factor_c / 2 * np.dot(self._state.a0.transpose(), self._state.a0) +\
                self._state.rho / 2 * np.eye(binary_size)
        # for i in range(binary_size):
        #     for j in range(i, binary_size):
        #         op1.objective.set_quadratic_coefficients(i, j, quadratic_objective[i, j])
        op1.objective.quadratic = quadratic_objective

        # prepare and set linear objective.
        linear_objective = self._state.c0 - \
            self._params.factor_c * np.dot(self._state.b0, self._state.a0) + \
            self._state.rho * (- self._state.y - self._state.z) + \
            self._state.lambda_mult

        # for i in range(binary_size):
        #     op1.objective.set_linear(i, linear_objective[i])

        op1.objective.linear = linear_objective
        return op1

    def _create_step2_problem(self) -> QuadraticProgram:
        """Creates a step 2 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op2 = QuadraticProgram()

        continuous_size = len(self._state.continuous_indices)
        binary_size = len(self._state.binary_indices)
        # lower_bounds = self._state.op.variables.get_lower_bounds(self._state.continuous_indices)
        # upper_bounds = self._state.op.variables.get_upper_bounds(self._state.continuous_indices)
        # if continuous_size:
        #     # add u variables.
        #     op2.variables.add(names=["u0_" + str(i + 1) for i in range(continuous_size)],
        #                       types=["C"] * continuous_size, lb=lower_bounds, ub=upper_bounds)
        continuous_index = 0
        for variable in self._state.op.variables:
            if variable.vartype == VarType.CONTINUOUS:
                op2.continuous_var(name="u0_" + str(continuous_index + 1),
                                   lowerbound=variable.lowerbound, upperbound=variable.upperbound)
                continuous_index += 1

        # add z variables.
        for i in range(binary_size):
            op2.continuous_var(name="z0_" + str(i + 1), lowerbound=0, upperbound=1.)

        # todo: do we need the 2*
        q_z = self._state.rho / 2 * np.eye(binary_size)
        op2.objective.quadratic = block_diag(self._state.q1, q_z)
        # set quadratic objective coefficients for u variables.
        # if continuous_size:
        #     q_u = self._state.q1
        #     for i in range(continuous_size):
        #         for j in range(i, continuous_size):
        #             op2.objective.set_quadratic_coefficients(i, j, q_u[i, j])

        # set quadratic objective coefficients for z variables.
        # NOTE: The multiplication by 2 is needed for the solvers to parse
        # the quadratic coefficients.
        # q_z = 2 * (self._state.rho / 2 * np.eye(binary_size))
        # for i in range(binary_size):
        #     for j in range(i, binary_size):
        #         op2.objective.set_quadratic_coefficients(i + continuous_size, j + continuous_size,
        #                                                  q_z[i, j])

        linear_z = -1 * self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.y)
        op2.objective.linear = np.concatenate((self._state.c1, linear_z))
        # set linear objective for u variables.
        # if continuous_size:
        #     linear_u = self._state.c1
        #     for i in range(continuous_size):
        #         op2.objective.set_linear(i, linear_u[i])

        # set linear objective for z variables.
        # linear_z = -1 * self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.y)
        # for i in range(binary_size):
        #     op2.objective.set_linear(i + continuous_size, linear_z[i])

        # constraints for z.
        # A1 z <= b1.
        constraint_count = self._state.a1.shape[0]
        # in SparsePair val="something from numpy" causes an exception
        # when saving a model via cplex method.
        # rhs="something from numpy" is ok.
        # so, we convert every single value to python float
        # lin_expr = [SparsePair(ind=list(range(continuous_size, continuous_size + binary_size)),
        #                        val=self._state.a1[i, :].tolist()) for i in
        #             range(constraint_count)]
        # op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count,
        #                            rhs=list(self._state.b1))

        for i in range(constraint_count):
            linear = np.concatenate((np.zeros(continuous_size), self._state.a1[i, :]))
            op2.linear_constraint(linear=linear, sense=ConstraintSense.LE, rhs=self._state.b1[i])

        # if continuous_size:
        #     # A2 z + A3 u <= b2
        #     constraint_count = self._state.a2.shape[0]
        #     lin_expr = [SparsePair(ind=list(range(continuous_size + binary_size)),
        #                            val=self._state.a3[i, :].tolist() +
        #                            self._state.a2[i, :].tolist())
        #                 for i in range(constraint_count)]
        #     op2.linear_constraints.add(lin_expr=lin_expr,
        #                                senses=["L"] * constraint_count,
        #                                rhs=self._state.b2.tolist())
        if continuous_size:
            # A2 z + A3 u <= b2
            constraint_count = self._state.a2.shape[0]
            for i in range(constraint_count):
                linear = np.concatenate((self._state.a3[i, :], self._state.a2[i, :]))
                op2.linear_constraint(linear=linear, sense=ConstraintSense.LE, rhs=self._state.b2[i])

        # if continuous_size:
        #     # A4 u <= b3
        #     constraint_count = self._state.a4.shape[0]
        #     lin_expr = [SparsePair(ind=list(range(continuous_size)),
        #                            val=self._state.a4[i, :].tolist()) for i in
        #                 range(constraint_count)]
        #     op2.linear_constraints.add(lin_expr=lin_expr,
        #                                senses=["L"] * constraint_count,
        #                                rhs=self._state.b3.tolist())

        if continuous_size:
            # A4 u <= b3
            constraint_count = self._state.a4.shape[0]
            for i in range(constraint_count):
                linear = np.concatenate((self._state.a4[i, :], np.zeros(binary_size)))
                op2.linear_constraint(linear=linear, sense=ConstraintSense.LE, rhs=self._state.b3[i])


        # add quadratic constraints for
        # In the step 2, we basically need to copy all quadratic constraints of the original
        # problem, and substitute binary variables with variables 0<=z<=1.
        # The quadratic constraint can have any equality/inequality sign.
        #
        #  For example:
        #  Original problem:
        #  Quadratic_constraint: x**2 + u**2 <= 1
        #  Vars: x binary, L <= u <= U
        #
        #  Step 2:
        #  Quadratic_constraint: z**2 + u**2 <= 1
        #  Vars: 0<=z<=1, L <= u <= U
        # for linear, quad, sense, rhs \
        #         in zip(self._state.op.quadratic_constraints.get_linear_components(),
        #                self._state.op.quadratic_constraints.get_quadratic_components(),
        #                self._state.op.quadratic_constraints.get_senses(),
        #                self._state.op.quadratic_constraints.get_rhs()):
        #     # types in the loop: SparsePair, SparseTriple, character, float
        #     print(linear, quad, sense, rhs)
        #     new_linear = SparsePair(ind=self._binary_indices_to_continuous(linear.ins),
        #                             val=linear.val)
        #     new_quadratic = SparseTriple(ind1=self._binary_indices_to_continuous(quad.ind1),
        #                                  ind2=self._binary_indices_to_continuous(quad.ind2),
        #                                  val=quad.val)
        #     op2.quadratic_constraints.add(lin_expr=new_linear, quad_expr=new_quadratic,
        #                                   sense=sense, rhs=rhs)

        return op2

    def _binary_indices_to_continuous(self, binary_indices: List[int]) -> List[int]:
        # todo: implement
        return binary_indices

    def _create_step3_problem(self) -> QuadraticProgram:
        """Creates a step 3 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op3 = QuadraticProgram()
        # add y variables.
        binary_size = len(self._state.binary_indices)
        # op3.variables.add(names=["y_" + str(i + 1) for i in range(binary_size)],
        #                   types=["C"] * binary_size, lb=[-np.inf] * binary_size,
        #                   ub=[np.inf] * binary_size)
        for i in range(binary_size):
            op3.continuous_var(name="y_" + str(i + 1), lowerbound=-np.inf, upperbound=np.inf)

        # set quadratic objective.
        # todo: do we need 2*?
        q_y = self._params.beta / 2 * np.eye(binary_size) + \
                   self._state.rho / 2 * np.eye(binary_size)
        # for i in range(binary_size):
        #     for j in range(i, binary_size):
        #         op3.objective.set_quadratic_coefficients(i, j, q_y[i, j])
        op3.objective.quadratic = q_y

        linear_y = - self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.z)
        # for i in range(binary_size):
        #     op3.objective.set_linear(i, linear_y[i])
        op3.objective.linear = linear_y

        return op3

    def _update_x0(self, op1: QuadraticProgram) -> np.ndarray:
        """Solves the Step1 QuadraticProgram via the qubo optimizer.

        Args:
            op1: the Step1 QuadraticProgram.

        Returns:
            A solution of the Step1, as a numpy array.
        """
        return np.asarray(self._qubo_optimizer.solve(op1).x)

    def _update_x1(self, op2: QuadraticProgram) -> (np.ndarray, np.ndarray):
        """Solves the Step2 QuadraticProgram via the continuous optimizer.

        Args:
            op2: the Step2 QuadraticProgram

        Returns:
            A solution of the Step2, as a pair of numpy arrays.
            First array contains the values of decision variables u, and
            second array contains the values of decision variables z.

        """
        vars_op2 = self._continuous_optimizer.solve(op2).x
        vars_u = np.asarray(vars_op2[:len(self._state.continuous_indices)])
        vars_z = np.asarray(vars_op2[len(self._state.continuous_indices):])
        return vars_u, vars_z

    def _update_y(self, op3: QuadraticProgram) -> np.ndarray:
        """Solves the Step3 QuadraticProgram via the continuous optimizer.

        Args:
            op3: the Step3 QuadraticProgram

        Returns:
            A solution of the Step3, as a numpy array.

        """
        return np.asarray(self._continuous_optimizer.solve(op3).x)

    def _get_best_merit_solution(self) -> (List[np.ndarray], float):
        """The ADMM solution is that for which the merit value is the best (least for min problems,
        greatest for max problems)
            * sol: Iterate with the best merit value
            * sol_val: Value of sol, according to the original objective

        Returns:
            A tuple of (sol, sol_val), where
                * sol: Solution with the best merit value
                * sol_val: Value of the objective function
        """

        it_best_merits = self._state.merits.index(
            self._state.sense * min(list(map(lambda x: self._state.sense * x, self._state.merits))))
        x_0 = self._state.x0_saved[it_best_merits]
        u_s = self._state.u_saved[it_best_merits]
        sol = [x_0, u_s]
        sol_val = self._state.cost_iterates[it_best_merits]
        return sol, sol_val

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
            * norm 1 of the body-rhs of the constraints A0 x0 - b0
            * -1 * min(body - rhs, 0) for geq constraints
            * max(body - rhs, 0) for leq constraints

        Returns:
            Violation of the constraints as a float value
        """

        cr0 = sum(np.abs(np.dot(self._state.a0, self._state.x0) - self._state.b0))

        eq1 = np.dot(self._state.a1, self._state.x0) - self._state.b1
        cr1 = sum(max(val, 0) for val in eq1)

        eq2 = np.dot(self._state.a2, self._state.x0) + np.dot(self._state.a3,
                                                              self._state.u) - self._state.b2
        cr2 = sum(max(val, 0) for val in eq2)

        return cr0 + cr1 + cr2

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

        def quadratic_form(matrix, x, c):
            return np.dot(x.T, np.dot(matrix, x)) + np.dot(c.T, x)

        obj_val = quadratic_form(self._state.q0, self._state.x0, self._state.c0)
        obj_val += quadratic_form(self._state.q1, self._state.u, self._state.c1)

        obj_val += self._state.op.objective.constant

        return obj_val

    def _get_solution_residuals(self, iteration: int) -> (float, float):
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

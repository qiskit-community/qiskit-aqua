import time
from typing import List

import numpy as np
from cplex import SparsePair

from qiskit.optimization.algorithms import CplexOptimizer
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationAlgorithm
from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from qiskit.optimization.problems.variables import CPX_BINARY, CPX_CONTINUOUS

from qiskit.optimization.results.optimization_result import OptimizationResult


class ADMMParameters:
    def __init__(self, rho_initial=10000, factor_c=100000, beta=1000, max_iter=10, tol=1.e-4, max_time=1800,
                 three_block=True, vary_rho=0, tau_incr=2, tau_decr=2, mu_res=10,
                 mu=1000, qubo_solver_class: OptimizationAlgorithm = CplexOptimizer,
                 continuous_solver_class: OptimizationAlgorithm = CplexOptimizer) -> None:
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
            If set to 0, then rho increases by 10% at each iteartion.
            If set to 1, then rho is modified according to primal and dual residuals.
            tau_incr: Parameter used in the rho update.
            tau_decr: Parameter used in the rho update.
            mu_res: Parameter used in the rho update.
            mu: Penalization for constraint residual. Used to compute the merit values.
            qubo_solver_class: A subclass of OptimizationAlgorithm that can effectively solve QUBO problems
            continuous_solver_class: A subclass of OptimizationAlgorithm that can solve continuous problems
        """
        super().__init__()
        self.mu = mu
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
        self.qubo_solver_class = qubo_solver_class
        self.continuous_solver_class = continuous_solver_class


class ADMMState:
    def __init__(self, binary_size: int, rho_initial: float) -> None:
        """
        Internal computation state of the ADMM implementation. Here, various variables are stored that are
        being updated during problem solving. The values are relevant to the problem being solved.
        The state is recreated for each optimization problem.

        Args:
            binary_size: Number of binary decision variables of the original problem
            rho_initial: Initial value of the rho parameter.
        """
        super().__init__()
        # These are the parameters that are updated in the ADMM iterations.
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


class ADMMOptimizer(OptimizationAlgorithm):
    def __init__(self, params: ADMMParameters = None) -> None:
        super().__init__()
        self._op = None
        self._binary_indices = []
        self._continuous_indices = []
        if params is None:
            # create default params
            params = ADMMParameters()
        # todo: consider keeping params as an object instead of copying
        self._three_block = params.three_block
        self._max_time = params.max_time
        self._tol = params.tol
        self._max_iter = params.max_iter
        self._factor_c = params.factor_c
        self._beta = params.beta
        self._mu_res = params.mu_res
        self._tau_decr = params.tau_decr
        self._tau_incr = params.tau_incr
        self._vary_rho = params.vary_rho
        self._three_block = params.three_block
        self._mu = params.mu
        self._rho_initial = params.rho_initial

        # note, we create instances of the solvers here instead of keeping classes
        self._qubo_solver = params.qubo_solver_class()
        self._continuous_solver = params.continuous_solver_class()

        # internal state where we'll keep intermediate solution
        # here, we just declare the class variable
        self._state = None

    def is_compatible(self, problem: OptimizationProblem):
        # 1. only binary and continuous variables are supported
        for var_index, var_type in enumerate(problem.variables.get_types()):
            if var_type != CPX_BINARY and var_type != CPX_CONTINUOUS:
                # var var_index is not binary and not continuous
                return False

        self._op = problem
        self._binary_indices = self._get_variable_indices(CPX_BINARY)
        self._continuous_indices = self._get_variable_indices(CPX_CONTINUOUS)

        # 2. binary and continuous variables are separable in objective
        for binary_index in self._binary_indices:
            for continuous_index in self._continuous_indices:
                coeff = problem.objective.get_quadratic_coefficients(binary_index, continuous_index)
                if coeff != 0:
                    # binary and continuous vars are mixed
                    return False

        # 3. no quadratic constraints are supported
        quad_constraints = problem.quadratic_constraints.get_num()
        if quad_constraints is not None and quad_constraints > 0:
            # quadratic constraints are not supported
            return False

        # todo: verify other properties of the problem
        return True

    def solve(self, problem: OptimizationProblem):
        """Tries to solves the given problem using ADMM algorithm.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem. Note that result.x it is a list [x0, u], with x0
            being the value of the binary variables in the ADMM solution, and u is the value of the continuous
            variables in the ADMM solution.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        self._op = problem

        # parse problem and convert to an ADMM specific representation
        self._binary_indices = self._get_variable_indices(CPX_BINARY)
        self._continuous_indices = self._get_variable_indices(CPX_CONTINUOUS)

        # create our computation state
        self._state = ADMMState(len(self._binary_indices), self._rho_initial)

        # debug
        # self.__dump_matrices_and_vectors()

        start_time = time.time()
        # we have not stated our computations yet, so elapsed time initialized as zero
        elapsed_time = 0

        it = 0
        r = 1.e+2

        # TODO: Handle objective sense. This has to be feed to the solvers of the subproblems.

        while (it < self._max_iter and r > self._tol) and (elapsed_time < self._max_time):

            op1 = self._create_step1_problem()
            # debug
            op1.write("op1.lp")

            self._state.x0 = self.update_x0(op1)
            # debug
            print("x0={}".format(self._state.x0))

            op2 = self._create_step2_problem()
            op2.write("op2.lp")

            self._state.u, self._state.z = self.update_x1(op2)
            # debug
            print("u={}".format(self._state.u))
            print("z={}".format(self._state.z))

            if self._three_block:
                op3 = self._create_step3_problem()
                op3.write("op3.lp")
                self._state.y = self.update_y(op3)
                # debug
                print("y={}".format(self._state.y))

            lambda_mult = self.update_lambda_mult()

            cost_iterate = self.get_cost_val()

            cr = self.get_cons_res()

            r, s = self.get_sol_res(it)

            merit = self.get_merit(cost_iterate, cr)
            # debug
            print("cost_iterate, cr, merit", cost_iterate, cr, merit)

            # costs and merits are saved with their original sign
            # TODO: obtain the sense, and update cost iterates and merits
            self._state.cost_iterates.append(cost_iterate)
            self._state.residuals.append(r)
            self._state.dual_residuals.append(s)
            self._state.cons_r.append(cr)
            self._state.merits.append(merit)
            self._state.lambdas.append(np.linalg.norm(lambda_mult))

            self._state.x0_saved.append(self._state.x0)
            self._state.u_saved.append(self._state.u)
            self._state.z_saved.append(self._state.z)
            self._state.z_saved.append(self._state.y)

            self.update_rho(r, s)

            it += 1
            elapsed_time = time.time() - start_time

        sol, sol_val = self.get_min_mer_sol()

        # third parameter is our internal state of computations
        result = OptimizationResult(sol, sol_val, self._state)
        # debug
        print("sol={0}, sol_val={1}".format(sol, sol_val))
        print("it {0}, state {1}".format(it, self._state))
        return result

    def _get_variable_indices(self, var_type: str) -> List[int]:
        indices = []
        for i, variable_type in enumerate(self._op.variables.get_types()):
            if variable_type == var_type:
                indices.append(i)

        return indices

    def get_q0(self):
        # TODO: Flip the sign, according to the optimization sense
        return self._get_q(self._binary_indices)

    def get_q1(self):
        # TODO: Flip the sign, according to the optimization sense
        return self._get_q(self._continuous_indices)

    def _get_q(self, variable_indices: List[int]) -> np.ndarray:
        size = len(variable_indices)
        q = np.zeros(shape=(size, size))
        # fill in the matrix
        # in fact we use re-indexed variables
        [q.itemset((i, j), self._op.objective.get_quadratic_coefficients(var_index_i, var_index_j))
         for i, var_index_i in enumerate(variable_indices)
         for j, var_index_j in enumerate(variable_indices)]
        return q

    def _get_c(self, variable_indices: List[int]) -> np.ndarray:
        c = np.array(self._op.objective.get_linear(variable_indices))
        return c

    def get_c0(self):
        # TODO: Flip the sign, according to the optimization sense
        return self._get_c(self._binary_indices)

    def get_c1(self):
        # TODO: Flip the sign, according to the optimization sense
        return self._get_c(self._continuous_indices)

    def _assign_row_values(self, matrix: List[List[float]], vector: List[float], constraint_index, variable_indices):
        # assign matrix row
        row = []
        [row.append(self._op.linear_constraints.get_coefficients(constraint_index, var_index))
         for var_index in variable_indices]
        matrix.append(row)

        # assign vector row
        vector.append(self._op.linear_constraints.get_rhs(constraint_index))

        # flip the sign if constraint is G, we want L constraints
        if self._op.linear_constraints.get_senses(constraint_index) == "G":
            # invert the sign to make constraint "L"
            matrix[-1] = [-1 * el for el in matrix[-1]]
            vector[-1] = -1 * vector[-1]

    def _create_ndarrays(self, matrix: List[List[float]], vector: List[float], size: int) -> (np.ndarray, np.ndarray):
        # if we don't have such constraints, return just dummy arrays
        if len(matrix) != 0:
            return np.array(matrix), np.array(vector)
        else:
            return np.array([0] * size).reshape((1, -1)), np.zeros(shape=(1,))

    def get_a0_b0(self) -> (np.ndarray, np.ndarray):
        matrix = []
        vector = []

        senses = self._op.linear_constraints.get_senses()
        index_set = set(self._binary_indices)
        for constraint_index, sense in enumerate(senses):
            # we check only equality constraints here
            if sense != "E":
                continue
            row = self._op.linear_constraints.get_rows(constraint_index)
            if set(row.ind).issubset(index_set):
                self._assign_row_values(matrix, vector, constraint_index, self._binary_indices)
            else:
                raise ValueError(
                    "Linear constraint with the 'E' sense must contain only binary variables, "
                    "row indices: {}, binary variable indices: {}".format(row, self._binary_indices))

        return self._create_ndarrays(matrix, vector, len(self._binary_indices))

    def _get_inequality_matrix_and_vector(self, variable_indices: List[int]) -> (List[List[float]], List[float]):
        # extracting matrix and vector from constraints like Ax <= b
        matrix = []
        vector = []
        senses = self._op.linear_constraints.get_senses()

        index_set = set(variable_indices)
        for constraint_index, sense in enumerate(senses):
            if sense == "E" or sense == "R":
                # TODO: Ranged constraints should be supported
                continue
            # sense either G or L
            row = self._op.linear_constraints.get_rows(constraint_index)
            if set(row.ind).issubset(index_set):
                self._assign_row_values(matrix, vector, constraint_index, variable_indices)

        return matrix, vector

    def get_a1_b1(self) -> (np.ndarray, np.ndarray):
        matrix, vector = self._get_inequality_matrix_and_vector(self._binary_indices)
        return self._create_ndarrays(matrix, vector, len(self._binary_indices))

    def get_a4_b3(self) -> (np.ndarray, np.ndarray):
        matrix, vector = self._get_inequality_matrix_and_vector(self._continuous_indices)

        return self._create_ndarrays(matrix, vector, len(self._continuous_indices))

    def get_a2_a3_b2(self) -> (np.ndarray, np.ndarray, np.ndarray):
        matrix = []
        vector = []
        senses = self._op.linear_constraints.get_senses()

        binary_index_set = set(self._binary_indices)
        continuous_index_set = set(self._continuous_indices)
        all_variables = self._binary_indices + self._continuous_indices
        for constraint_index, sense in enumerate(senses):
            if sense == "E" or sense == "R":
                # TODO: Ranged constraints should be supported as well
                continue
            # sense either G or L
            row = self._op.linear_constraints.get_rows(constraint_index)
            row_indices = set(row.ind)
            # we must have a least one binary and one continuous variable, otherwise it is another type of constraints
            if len(row_indices & binary_index_set) != 0 and len(row_indices & continuous_index_set) != 0:
                self._assign_row_values(matrix, vector, constraint_index, all_variables)

        matrix, b2 = self._create_ndarrays(matrix, vector, len(all_variables))
        # a2
        a2 = matrix[:, 0:len(self._binary_indices)]
        a3 = matrix[:, len(self._binary_indices):]
        return a2, a3, b2

    def _create_step1_problem(self):
        op1 = OptimizationProblem()

        binary_size = len(self._binary_indices)
        # create the same binary variables
        # op1.variables.add(names=["x0_" + str(i + 1) for i in range(binary_size)], types=["B"] * binary_size)
        op1.variables.add(names=["x0_" + str(i + 1) for i in range(binary_size)],
                          types=["I"] * binary_size,
                          lb=[0.] * binary_size,
                          ub=[1.] * binary_size)

        # prepare and set quadratic objective.
        # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        a0, b0 = self.get_a0_b0()
        quadratic_objective = 2 * (
                self.get_q0() + self._factor_c / 2 * np.dot(a0.transpose(), a0) +
                self._state.rho / 2 * np.eye(binary_size)
        )
        for i in range(binary_size):
            for j in range(i, binary_size):
                op1.objective.set_quadratic_coefficients(i, j, quadratic_objective[i, j])
                op1.objective.set_quadratic_coefficients(j, i, quadratic_objective[i, j])

        # prepare and set linear objective
        c0 = self.get_c0()
        linear_objective = c0 - self._factor_c * np.dot(b0, a0) + self._state.rho * (self._state.y - self._state.z)
        for i in range(binary_size):
            op1.objective.set_linear(i, linear_objective[i])
        return op1

    def _create_step2_problem(self):
        op2 = OptimizationProblem()

        continuous_size = len(self._continuous_indices)
        binary_size = len(self._binary_indices)
        lb = self._op.variables.get_lower_bounds(self._continuous_indices)
        ub = self._op.variables.get_upper_bounds(self._continuous_indices)
        if continuous_size:
            # add u variables
            op2.variables.add(names=["u0_" + str(i + 1) for i in range(continuous_size)],
                              types=["C"] * continuous_size, lb=lb, ub=ub)

        # add z variables
        op2.variables.add(names=["z0_" + str(i + 1) for i in range(binary_size)],
                          types=["C"] * binary_size,
                          lb=[0.] * binary_size,
                          ub=[1.] * binary_size)

        # set quadratic objective coefficients for u variables
        if continuous_size:
            # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
            q_u = 2 * (self.get_q1())
            for i in range(continuous_size):
                for j in range(i, continuous_size):
                    # todo: verify that we don't need both calls
                    op2.objective.set_quadratic_coefficients(i, j, q_u[i, j])
                    op2.objective.set_quadratic_coefficients(j, i, q_u[i, j])

        # set quadratic objective coefficients for z variables.
        # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        q_z = 2 * (self._state.rho / 2 * np.eye(binary_size))
        for i in range(binary_size):
            for j in range(i, binary_size):
                # todo: verify that we don't need both calls
                op2.objective.set_quadratic_coefficients(i + continuous_size, j + continuous_size, q_z[i, j])
                op2.objective.set_quadratic_coefficients(j + continuous_size, i + continuous_size, q_z[i, j])

        # set linear objective for u variables
        if continuous_size:
            linear_u = self.get_c1()
            for i in range(continuous_size):
                op2.objective.set_linear(i, linear_u[i])

        # set linear objective for z variables
        linear_z = -1 * self._state.lambda_mult - self._state.rho * (self._state.x0 + self._state.y)
        for i in range(binary_size):
            op2.objective.set_linear(i + continuous_size, linear_z[i])

        # constraints for z
        # A1 z <= b1
        a1, b1 = self.get_a1_b1()
        constraint_count = a1.shape[0]
        # in SparsePair val="something from numpy" causes an exception when saving a model via cplex method.
        # rhs="something from numpy" is ok
        # so, we convert every single value to python float, todo: consider removing this conversion
        lin_expr = [SparsePair(ind=list(range(continuous_size, continuous_size + binary_size)),
                               val=self._to_list(a1[i, :])) for i in range(constraint_count)]
        op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=list(b1))

        if continuous_size:
            # A2 z + A3 u <= b2
            a2, a3, b2 = self.get_a2_a3_b2()
            constraint_count = a2.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size + binary_size)),
                                   val=self._to_list(a3[i, :]) + self._to_list(a2[i, :]))
                        for i in range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=self._to_list(b2))

        if continuous_size:
            # A4 u <= b3
            a4, b3 = self.get_a4_b3()
            constraint_count = a4.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size)),
                                   val=self._to_list(a4[i, :])) for i in range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=self._to_list(b3))

        return op2

    def _create_step3_problem(self):
        op3 = OptimizationProblem()
        # add y variables
        binary_size = len(self._binary_indices)
        op3.variables.add(names=["y_" + str(i + 1) for i in range(binary_size)],
                          types=["C"] * binary_size)

        # set quadratic objective.
        # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        q_y = 2 * (self._beta / 2 * np.eye(binary_size) + self._state.rho / 2 * np.eye(binary_size))
        for i in range(binary_size):
            for j in range(i, binary_size):
                op3.objective.set_quadratic_coefficients(i, j, q_y[i, j])
                op3.objective.set_quadratic_coefficients(j, i, q_y[i, j])

        linear_y = self._state.lambda_mult + self._state.rho * (self._state.x0 - self._state.z)
        for i in range(binary_size):
            op3.objective.set_linear(i, linear_y[i])

        return op3

    # when a plain list() call is used a numpy type of values makes cplex to fail when cplex.write() is called.
    # for debug only, list() should be used instead
    def _to_list(self, values):
        out_list = []
        for el in values:
            out_list.append(float(el))
        return out_list

    def update_x0(self, op1: OptimizationProblem) -> np.ndarray:
        # TODO: Check output type of qubo_solver.solve(op1).x
        return np.asarray(self._qubo_solver.solve(op1).x)

    def update_x1(self, op2: OptimizationProblem) -> (np.ndarray, np.ndarray):
        vars_op2 = self._continuous_solver.solve(op2).x
        # TODO: Check output type
        u = np.asarray(vars_op2[:len(self._continuous_indices)])
        z = np.asarray(vars_op2[len(self._continuous_indices):])
        return u, z

    def update_y(self, op3):
        # TODO: Check output type
        return np.asarray(self._continuous_solver.solve(op3).x)

    def get_min_mer_sol(self):
        """
        The ADMM solution is that for which the merit value is the least
            * sol: Iterate with the least merit value
            * sol_val: Value of sol, according to the original objective

        Returns:
            A tuple of (sol, sol_val), where
                * sol: Iterate with the least merit value
                * sol_val: Value of sol, according to the original objective
        """
        it_min_merits = self._state.merits.index(min(self._state.merits))
        x0 = self._state.x0_saved[it_min_merits]
        u = self._state.u_saved[it_min_merits]
        sol = [x0, u]
        sol_val = self._state.cost_iterates[it_min_merits]
        return sol, sol_val
    
    def update_lambda_mult(self):
        return self._state.lambda_mult + self._state.rho * (self._state.x0 - self._state.z + self._state.y)
    
    def update_rho(self, r, s):
        """
        Updating the rho parameter in ADMM

        Args:
            r: primal residual
            s: dual residual
        """

        if self._vary_rho == 0:
            # Increase rho, to aid convergence.
            if self._state.rho < 1.e+10:
                self._state.rho *= 1.1
        elif self._vary_rho == 1:
            if r > self._mu_res * s:
                self._state.rho = self._tau_incr * self._state.rho
            elif s > self._mu_res * r:
                self._state.rho = self._tau_decr * self._state.rho

    def get_cons_res(self):
        """
        Compute violation of the constraints of the original problem, as:
            * norm 1 of the body-rhs of the constraints A0 x0 - b0
            * -1 * min(body - rhs, 0) for geq constraints
            * max(body - rhs, 0) for leq constraints
        """

        # TODO: think whether a0, b0 should be saved somewhere.. Might move to state?
        a0, b0 = self.get_a0_b0()
        cr0 = sum(np.abs(np.dot(a0, self._state.x0) - b0))

        a1, b1 = self.get_a1_b1()
        eq1 = np.dot(a1, self._state.x0) - b1
        cr1 = sum(max(val, 0) for val in eq1)

        a2, a3, b2 = self.get_a2_a3_b2()
        eq2 = np.dot(a2, self._state.x0) + np.dot(a3, self._state.u) - b2
        cr2 = sum(max(val, 0) for val in eq2)

        return cr0+cr1+cr2

    def get_merit(self, cost_iterate, cr):
        """
        Compute merit value associated with the current iterate
        """
        return cost_iterate + self._mu * cr

    def get_cost_val(self):
        """
        Computes the value of the objective function.
        """
        # quadr_form = lambda A, x, c: np.dot(x.T, np.dot(A, x)) + np.dot(c.T, x)
        def quadratic_form(matrix, x, c): return np.dot(x.T, np.dot(matrix, x)) + np.dot(c.T, x)

        q0 = self.get_q0()
        q1 = self.get_q1()
        c0 = self.get_c0()
        c1 = self.get_c1()

        obj_val = quadratic_form(q0, self._state.x0, c0)
        obj_val += quadratic_form(q1, self._state.u, c1)

        return obj_val

    def get_sol_res(self, it):
        """
        Compute primal and dual residual.

        Args:
            it:
        """
        elements = self._state.x0 - self._state.z - self._state.y
        # debug
        # elements = np.asarray([x0[i] - z[i] + y[i] for i in self.range_x0_vars])
        r = pow(sum(e ** 2 for e in elements), 0.5)
        if it > 0:
            elements_dual = self._state.z - self._state.z_saved[it-1]
        else:
            elements_dual = self._state.z - self._state.z_init
        # debug
        # elements_dual = np.asarray([z[i] - z_old[i] for i in self.range_x0_vars])
        s = self._state.rho * pow(sum(e ** 2 for e in elements_dual), 0.5)

        return r, s

    # only for debugging!
    def __dump_matrices_and_vectors(self):
        print("In admm_optimizer.py")
        q0 = self.get_q0()
        print("Q0")
        print(q0)
        print("Q0 shape")
        print(q0.shape)
        q1 = self.get_q1()
        print("Q1")
        print(q1)
        print("Q1")
        print(q1.shape)

        c0 = self.get_c0()
        print("c0")
        print(c0)
        print("c0 shape")
        print(c0.shape)
        c1 = self.get_c1()
        print("c1")
        print(c1)
        print("c1 shape")
        print(c1.shape)

        a0, b0 = self.get_a0_b0()
        print("A0")
        print(a0)
        print("A0")
        print(a0.shape)
        print("b0")
        print(b0)
        print("b0 shape")
        print(b0.shape)

        a1, b1 = self.get_a1_b1()
        print("A1")
        print(a1)
        print("A1 shape")
        print(a1.shape)
        print("b1")
        print(b1)
        print("b1 shape")
        print(b1.shape)

        a4, b3 = self.get_a4_b3()
        print("A4")
        print(a4)
        print("A4 shape")
        print(a4.shape)
        print("b3")
        print(b3)
        print("b3 shape")
        print(b3.shape)

        a2, a3, b2 = self.get_a2_a3_b2()
        print("A2")
        print(a2)
        print("A2 shape")
        print(a2.shape)
        print("A3")
        print(a3)
        print("A3")
        print(a3.shape)
        print("b2")
        print(b2)
        print("b2 shape")
        print(b2.shape)

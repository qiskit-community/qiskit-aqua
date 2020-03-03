from typing import List

import numpy as np
from cplex import SparsePair
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationAlgorithm
from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from qiskit.optimization.problems.variables import CPX_BINARY, CPX_CONTINUOUS

from qiskit.optimization.results.optimization_result import OptimizationResult


class ADMMParameters:
    def __init__(self, rho=10000, factor_c=100000, beta=1000) -> None:
        """
        Defines parameter for ADMM.
        :param rho: Rho parameter of ADMM.
        :param factor_c: Penalizing factor for equality constraints, when mapping to QUBO.
        :param beta: Penalization for y decision variables.
        """
        super().__init__()
        self.factor_c = factor_c
        self.rho = rho
        self.beta = beta


class ADMMState:
    def __init__(self, binary_size: int) -> None:
        super().__init__()
        self.y = np.zeros(binary_size)
        self.z = np.zeros(binary_size)
        self.lambda_mult = np.zeros(binary_size)
        self.x0 = np.zeros(binary_size)


class ADMMOptimizer(OptimizationAlgorithm):
    def __init__(self, params: ADMMParameters = None) -> None:
        super().__init__()
        self._op = None
        self._binary_indices = []
        self._continuous_indices = []
        if params is None:
            # create default params
            params = ADMMParameters()
        # todo: keep parameters as ADMMParameters or copy to the class level?
        self._factor_c = params.factor_c
        self._rho = params.rho
        self._beta = params.beta

        # internal state where we'll keep intermediate solution
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
        self._op = problem

        # parse problem and convert to an ADMM specific representation
        self._binary_indices = self._get_variable_indices(CPX_BINARY)
        self._continuous_indices = self._get_variable_indices(CPX_CONTINUOUS)

        # create our computation state
        self._state = ADMMState(len(self._binary_indices))

        # debug
        # self.__dump_matrices_and_vectors()

        op1 = self._create_step1_problem()
        # debug
        op1.write("op1.lp")

        op2 = self._create_step2_problem()
        op2.write("op2.lp")

        op3 = self._create_step3_problem()
        op3.write("op3.lp")

        # solve the problem
        # ...
        # prepare the solution

        # actual results
        x = 0
        # function value
        fval = 0
        # third parameter is our internal state of computations
        result = OptimizationResult(x, fval, self._state)
        return result

    def _get_variable_indices(self, var_type: str) -> List[int]:
        indices = []
        for i, variable_type in enumerate(self._op.variables.get_types()):
            if variable_type == var_type:
                indices.append(i)

        return indices

    def get_q0(self):
        return self._get_q(self._binary_indices)

    def get_q1(self):
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
        return self._get_c(self._binary_indices)

    def get_c1(self):
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

        # prepare and set quadratic objective. NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        a0, b0 = self.get_a0_b0()
        quadratic_objective = 2 * (
                self.get_q0() + self._factor_c / 2 * np.dot(a0.transpose(), a0) +
                self._rho / 2 * np.eye(binary_size)
        )
        for i in range(binary_size):
            for j in range(i, binary_size):
                op1.objective.set_quadratic_coefficients(i, j, quadratic_objective[i, j])
                op1.objective.set_quadratic_coefficients(j, i, quadratic_objective[i, j])

        # prepare and set linear objective
        c0 = self.get_c0()
        linear_objective = c0 - self._factor_c * np.dot(b0, a0) + self._rho * (self._state.y - self._state.z)
        for i in range(binary_size):
            op1.objective.set_linear(i, linear_objective[i])
        return op1

    def _create_step2_problem(self):
        op2 = OptimizationProblem()

        continuous_size = len(self._continuous_indices)
        binary_size = len(self._binary_indices)
        lb = self._op.variables.get_lower_bounds(self._binary_indices)
        ub = self._op.variables.get_upper_bounds(self._binary_indices)
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
            q_u = 2 * (self.get_q1()) #NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
            for i in range(continuous_size):
                for j in range(i, continuous_size):
                    op2.objective.set_quadratic_coefficients(i, j, q_u[i, j])
                    op2.objective.set_quadratic_coefficients(j, i, q_u[i, j])

        # set quadratic objective coefficients for z variables. 
        q_z = 2 * (self._rho / 2 * np.eye(binary_size)) # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        for i in range(binary_size):
            for j in range(i, binary_size):
                op2.objective.set_quadratic_coefficients(i + continuous_size, j + continuous_size, q_z[i, j])
                op2.objective.set_quadratic_coefficients(j + continuous_size, i + continuous_size, q_z[i, j])

        # set linear objective for u variables
        if continuous_size:
            linear_u = self.get_c1()
            for i in range(continuous_size):
                op2.objective.set_linear(i, linear_u[i])

        # set linear objective for z variables
        linear_z = -1 * self._state.lambda_mult - self._rho * (self._state.x0 + self._state.y)
        for i in range(binary_size):
            op2.objective.set_linear(i + continuous_size, linear_z[i])

        # constraints for z
        # A1 z <= b1
        a1, b1 = self.get_a1_b1()
        constraint_count = a1.shape[0]
        # in SparsePair val="something from numpy" causes an exception when saving a model via cplex method. rhs="something from numpy" is ok
        # so, we convert every single value to python float, todo: consider removing this conversion
        lin_expr = [SparsePair(ind=list(range(continuous_size, continuous_size + binary_size)), val=self._to_list(a1[i, :])) for i in range(constraint_count)]
        op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=list(b1))

        if continuous_size:
            # A2 z + A3 u <= b2
            a2, a3, b2 = self.get_a2_a3_b2()
            constraint_count = a2.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size + binary_size)), val=self._to_list(a3[i, :]) + self._to_list(a2[i, :])) for i in range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=self._to_list(b2))

        if continuous_size:
            # A4 u <= b3
            a4, b3 = self.get_a4_b3()
            constraint_count = a4.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size)), val=self._to_list(a4[i, :])) for i in range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count, rhs=self._to_list(b3))

        # todo: do we keep u bounds, z bounds as bounds or as constraints. I would keep bounds as bounds.

        return op2

    def _create_step3_problem(self):
        op3 = OptimizationProblem()
        # add y variables
        binary_size = len(self._binary_indices)
        op3.variables.add(names=["y_" + str(i + 1) for i in range(binary_size)],
                          types=["C"] * binary_size)

        # set quadratic objective. NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coefficients.
        q_y = 2 * (self._beta / 2 * np.eye(binary_size) + self._rho / 2 * np.eye(binary_size))
        for i in range(binary_size):
            for j in range(i, binary_size):
                op3.objective.set_quadratic_coefficients(i, j, q_y[i, j])
                op3.objective.set_quadratic_coefficients(j, i, q_y[i, j])

        linear_y = self._state.lambda_mult + self._rho * (self._state.x0 - self._state.z)
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

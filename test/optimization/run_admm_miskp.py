"""
Created: 2020-01-24
@author Claudio Gambella [claudio.gambella1@ie.ibm.com]

This solves Mixed-Integer Setup Knapsack Problem via ADMM
References: Exact and heuristic solution approaches for the mixed integer setup knapsack problem, Altay et. al, EJOR, 2008.
Gambella, C., & Simonetto, A. (2020). Multi-block ADMM Heuristics for Mixed-Binary Optimization on Classical 
and Quantum Computers. arXiv preprint arXiv:2001.02069.
"""
import os
import sys
import time
import numpy as np
from qiskit import BasicAer

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import COBYLA

from qiskit.optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMOptimizer, ADMMParameters
from qiskit.optimization.problems import OptimizationProblem


try:
    import cplex
    from cplex.exceptions import CplexError
except ImportError:
    print("Failed to import cplex.")
    sys.exit(1)
from cplex import SparsePair

def create_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
def nm(stem, index1, index2=None, index3=None):
    """A method to return a string representing the name of a decision variable or a constraint, given its indices.
        Attributes:
        stem: Element name.
        index1, index2, index3: Element indices.
    """
    if index2==None: return stem + "(" + str(index1) + ")"
    if index3==None: return stem + "(" + str(index1) + "," + str(index2) + ")"
    return stem + "(" + str(index1) + "," + str(index2) + "," + str(index3) + ")"

def __dump_arrays(Q0, Q1, c0, c1, A0, b0, A1, b1, A4, b3, A2, A3, b2):
    """
    For debugging purposes
    :param Q0: 
    :param Q1: 
    :param c0: 
    :param c1: 
    :param A0: 
    :param b0: 
    :param A1: 
    :param b1: 
    :param A4: 
    :param b3: 
    :param A2: 
    :param A3: 
    :param b2: 
    :return: 
    """
    print("Claudio's implementation")
    print("Q0")
    print(Q0)
    print("Q0 shape")
    print(Q0.shape)
    print("Q1")
    print(Q1)
    print("Q1 shape")
    print(Q1.shape)

    print("c0")
    print(c0)
    print("c0 shape")
    print(c0.shape)
    print("c1")
    print(c1)
    print("c1 shape")
    print(c1.shape)

    print("A0")
    print(A0)
    print("A0 shape")
    print(A0.shape)
    print("b0")
    print(b0)
    print("b0 shape")
    print(b0.shape)

    print("A1")
    print(A1)
    print("A1 shape")
    print(A1.shape)
    print("b1")
    print(b1)
    print("b1 shape")
    print(b1.shape)

    print("A4")
    print(A4)
    print("A4 shape")
    print(A4.shape)
    print("b3")
    print(b3)
    print("b3 shape")
    print(b3.shape)

    print("A2")
    print(A2)
    print("A2 shape")
    print(A2.shape)
    print("A3")
    print(A3)
    print("A3 shape")
    print(A3.shape)
    print("b2")
    print(b2)
    print("b2 shape")
    print(b2.shape)

    print("End of Claudio's implementation")


def get_instance_params():
    """
    Get parameters for a  Mixed Integer Setup Knapsack Problem (MISKP) instance.

    :return:
    """
    #

    K = 2
    T = 10
    P = 45.10
    S = np.asarray([75.61, 75.54])

    D = np.asarray([9.78, 8.81, 9.08, 2.03, 8.9, 9, 4.12, 8.16, 6.55, 4.84, 3.78, 5.13,
                    4.72, 7.6, 2.83, 1.44, 2.45, 2.24, 6.3, 5.02]).reshape((K, T))

    C = np.asarray([-11.78, -10.81, -11.08, -4.03, -10.90, -11.00, -6.12, -10.16, -8.55, -6.84,
                    -5.78, -7.13, -6.72, -9.60, -4.83, -3.44, -4.45, -4.24, -8.30, -7.02]).reshape((K, T))

    n = K  # number of binary dec vars x0 (and z)
    m = K * T  # number of continuous dec vars u
    # size of x1 is then n+m

    # Parameters of the Quadratic Programming problem in the standard form for ADMM.
    ## Objective function
    Q0 = np.zeros((n, n))
    c0 = S.reshape(n)

    A0 = np.zeros((1, n))  # we set here one dummy equality constraint
    b0 = np.zeros(1)  # we set here one dummy equality constraint

    Q1 = np.zeros((m, m))
    c1 = C.reshape(m)

    ## Constraints
    # we set here one dummy A1 x0 \leq b1 constraint
    A1 = np.zeros((1, n))
    b1 = np.zeros(1)

    # A_2 z + A_3 u \leq b_2 -- > - y_k <= x_{k,t}
    A2 = np.zeros((m, n))
    for i in range(K):
        A2[T * i:T * (i + 1), i] = - np.ones(T)
    A3 = np.eye(m)
    b2 = np.zeros(m)

    # A_4 u <= b_3 --> sum_k sum_t D_{kt} x_{kt} \leq P, -x_{kt} \leq 0
    block1 = D.reshape((1, m))
    block2 = -np.eye(m)
    A4 = np.block([[block1], [block2]])
    b3 = np.hstack([P, np.zeros(m)])

    __dump_arrays(Q0, Q1, c0, c1, A0, b0, A1, b1, A4, b3, A2, A3, b2)

    return K, T, P, S, D, C



class Miskp:

    def __init__(self, K, T, P, S, D: np.ndarray, C: np.ndarray, verbose=False, relativeGap=0.0,
                 pairwise_incomp=0, multiple_choice=0):
        """
        Constructor method of the class.
        :param K: number of families
        :param T: number of items in each family
        :param C: value of including item t in family k in the knapsack
        :param D: resources consumed if item t in family k is included in the knapsack
        :param S: setup cost to include family k in the knapsack
        :param P: capacity of the knapsack
        :param verbose:
        :param relativeGap:
        """

        self.multiple_choice = multiple_choice
        self.pairwise_incomp = pairwise_incomp
        self.P = P
        self.S = S
        self.D = D
        self.C = C
        self.T = T
        self.K = K
        self.relativeGap = relativeGap
        self.verbose = verbose
        self.lp_folder = "./lps/"

    def create_params(self):
        self.range_K = range(self.K)
        self.range_T = range(self.T)

        # make sure instance params are floats

        self.S = [float(val) for val in self.S]
        self.C = self.C.astype(float)
        self.D = self.D.astype(float)

        self.n_x_vars = self.K * self.T
        self.n_y_vars = self.K

        self.range_x_vars = [(k, t) for k in self.range_K for t in self.range_T]
        self.range_y_vars = self.range_K


    def create_vars(self):

        self.op.variables.add(
            lb=[0.0] * self.n_x_vars,
            names=[nm("x", i, j) for i,j in self.range_x_vars])

        self.op.variables.add(
            # lb=[0.0] * self.n_y_vars,
            # ub=[1.0] * self.n_y_vars,
            types=["B"] * self.n_y_vars,
            names=[nm("y", i) for i in self.range_y_vars])

    def create_cons_capacity(self):
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[
                        nm("x", i, j) for i,j in self.range_x_vars
                    ]
                    ,
                    val=[self.D[i,j] for i,j in self.range_x_vars])
            ],
            senses="L",
            rhs=[self.P],
            names=["CAPACITY"])

    def create_cons_allocation(self):
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[nm("x", k, t)]+[nm("y", k)],
                    val=[1.0, -1.0])
                for k,t in self.range_x_vars
            ],
            senses="L" * self.n_x_vars,
            rhs=[0.0] * self.n_x_vars,
            names=[nm("ALLOCATION", k, t) for k, t in self.range_x_vars])


    def create_cons(self):
        """Method to populate the constraints"""

        self.create_cons_capacity()
        self.create_cons_allocation()

    def create_obj(self):
        self.op.objective.set_linear([(nm("y", k), self.S[k]) for k in self.range_K] +
                                     [(nm("x", k, t), self.C[k, t]) for k, t in self.range_x_vars]
                                     )

    def run_cplex_api(self):
        """Main method, which populates and solve the mathematical model via Python Cplex API

        """

        # Creation of the Cplex object
        self.op = OptimizationProblem()


        self.create_params()
        self.create_vars()
        self.create_obj()
        start_time = time.time()
        self.create_cons()
        constraints_time = time.time() - start_time
        if self.verbose:
            print ("Time to populate constraints:", constraints_time)

        # Save the model
        create_folder(self.lp_folder)

        self.op.write(self.lp_folder + "miskp.lp")

        out = 0
        return out

    def run_op(self):
        """Main method, which populates and solve the OptimizationProblem model via OptimizationAlgorithm

        """

        self.op = OptimizationProblem()

        self.create_params()
        self.create_vars()
        self.create_obj()
        self.create_cons()

        # Save the model
        create_folder(self.lp_folder)

        self.op.write(self.lp_folder + "miskp.lp")

        solver = ADMMOptimizer()
        solution = solver.solve(self.op)

        return solution
    
    def run_op_eigens(self):
        """Main method, which populates and solve the OptimizationProblem model via OptimizationAlgorithm

        """

        self.op = OptimizationProblem()

        self.create_params()
        self.create_vars()
        self.create_obj()
        self.create_cons()

        # Save the model
        create_folder(self.lp_folder)

        self.op.write(self.lp_folder + "miskp.lp")

        # QAOA
        # optimizer = COBYLA()
        # min_eigen_solver = QAOA(optimizer=optimizer)
        # qubo_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        # Note: a backend needs to be given, otherwise an error is raised in the _run method of VQE.
        # backend = 'statevector_simulator'
        # # backend = 'qasm_simulator'
        # min_eigen_solver.quantum_instance = BasicAer.get_backend(backend)

        # use numpy exact diagonalization
        qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())

        # Cplex
        # qubo_optimizer = CplexOptimizer()

        continuous_optimizer = CplexOptimizer()

        admm_params = ADMMParameters(qubo_optimizer=qubo_optimizer, continuous_optimizer=continuous_optimizer)

        solver = ADMMOptimizer(params=admm_params)
        solution = solver.solve(self.op)

        return solution


def toy_cplex_api():

    K, T, P, S, D, C = get_instance_params()
    pb = Miskp(K, T, P, S, D, C)
    result = pb.run_cplex_api()

def toy_op():

    K, T, P, S, D, C = get_instance_params()
    pb = Miskp(K, T, P, S, D, C)
    result = pb.run_op()
    
def toy_op_eigens():

    K, T, P, S, D, C = get_instance_params()
    pb = Miskp(K, T, P, S, D, C)
    result = pb.run_op_eigens()
    # debug
    print("results")
    print("x={}".format(result.x))
    print("fval={}".format(result.fval))
    # print("x0_saved={}".format(result.results.x0_saved))
    # print("u_saved={}".format(result.results.u_saved))
    # print("z_saved={}".format(result.results.z_saved))

if __name__ == '__main__':
    # toy_cplex_api()
    # toy_op()
    toy_op_eigens()







"""
Created: 2020-02-19
@author Claudio Gambella [claudio.gambella1@ie.ibm.com]
"""
import math
import os
import sys
import numpy as np

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
    - Populate a Bin Packing Problem (BPP) instance.
    - Create the instance parameters needed to apply ADMM.
    NOTE: The x decision variables are thought of as array parameters.

    n number of binary dec vars x0 (and z)
    m number of continuous dec vars u

    :return: n, m, Q0, c0, Q1, c1, A0, b0, A1, b1, A2, A3, b2, lb, ub ADMM parameters
    """
    #

    n_items = 2
    n_bins = n_items
    n_x_vars = n_items * n_bins
    C = 40
    w = [35, 31]

    n = n_x_vars + n_bins  # number of binary dec vars x0 (and z)
    m = 0  # number of continuous dec vars u

    # Parameters of the Quadratic Programming problem in the standard form for ADMM.
    Q0 = np.zeros((n, n))
    c0 = np.hstack((np.zeros(n_x_vars), np.ones(n_bins)))

    # A0 x0 = b0 --> \sum _i x_{ij} = 1
    A0 = np.tile(np.eye(n_items), n_bins)
    A0 = np.hstack((A0, np.zeros((n_items, n_items
                                 ))))
    b0 = np.ones(n_items)

    Q1 = np.zeros((m, m))
    c1 = np.zeros(m)

    # A1 z \leq b1
    A1 = np.zeros((n_bins, n))
    block1 = np.zeros((n_bins, n_x_vars))
    for i in range(n_bins):
        block1[i, i*n_items:(i+1)*n_items] = np.asarray(w)
    A1 = np.hstack((block1, -C * np.eye(n_bins)))
    b1 = np.zeros(n_bins)

    dummy_dim = 1
    A2 = np.zeros((dummy_dim, n))
    A3 = np.zeros((dummy_dim, m))
    b2 = np.zeros(dummy_dim)

    A4 = np.zeros((1,m))
    b3 = np.zeros(1)

    __dump_arrays(Q0, Q1, c0, c1, A0, b0, A1, b1, A4, b3, A2, A3, b2)

    return n_items, n_bins, C, w

class Bpp:

    def __init__(self, n_items, n_bins, C, w, verbose=False, relativeGap=0.0):
        """
        Constructor method of the class.
        :param n: number of items
        :param m: number of bins
        :param C: capacity of the bins
        :param w: weight of items
        :param verbose:
        :param relativeGap:
        """
        self.C = C
        self.relativeGap = relativeGap
        self.verbose = verbose
        self.w = w
        self.n = n_items
        self.m = n_bins
        self.lp_folder = "./outputs/lps/bpp/"

    def create_params(self, save_vars=False):
        self.range_n = range(self.n)
        self.range_m = range(self.m)
        self.range_mn = [(i,j) for i in self.range_m for j in self.range_n]
        self.nm = self.n*self.m

        # make sure instance params are floats
        self.w = [float(item) for item in self.w]
        self.C = float(self.C)

        if save_vars:
            self.l = self.get_lb_bins()
            self.n_x = self.m*self.n-self.m
            self.n_y = self.m-self.l

            self.range_x_vars = [(i, j) for i in self.range_m for j in self.range_n if
                                 j != 0]  # item j=0 is automatically assigned to bin 0
            self.range_y_vars = range(self.l, self.m)  # y_0, ..., y_{l-1} are not needed
        else:
            self.l = 0

            self.n_x = self.m * self.n
            self.n_y = self.m

            self.range_x_vars = self.range_mn
            self.range_y_vars = self.range_m

    def get_lb_bins(self) -> int:
        """
        Return lower bound on the number of bins needed, (see, e.g., Martello and Toth 1990)
        :return:
        """
        return math.ceil(sum(self.w)/self.C)


    def create_vars(self, x_var=True):
        """

        :param x_var:
        :return:
        """

        if x_var:
            self.op.variables.add(
                types=["B"] * self.n_x,
                names=[nm("x", i, j) for i,j in self.range_x_vars])

            self.op.variables.add(
                types=["B"] * self.n_y,
                names=[nm("y", i) for i in self.range_y_vars])


    def create_cons_eq(self, save_vars=False):

        if save_vars:
            self.items_to_assign = range(1, self.n)
        else:
            self.items_to_assign = self.range_n

        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[
                        nm("x", i, j) for i in self.range_m
                    ],
                    val=[1.0] * self.m)
                for j in self.items_to_assign
            ],
            senses="E" * len(self.items_to_assign),
            rhs=[1.0] * len(self.items_to_assign),
            names=[nm("ASSIGN_ITEM", j) for j in self.items_to_assign])

    def create_cons_ineq(self, save_vars=False):

        if not save_vars:
            self.op.linear_constraints.add(
                lin_expr=[
                    SparsePair(
                        ind=[
                            nm("x", i, j) for j in self.range_n
                        ] +[nm("y", i)]
                        ,
                        val=self.w + [-self.C])
                    for i in self.range_m
                ],
                senses="L" * self.m,
                rhs=[0.0] * self.m,
                names=[nm("CAPACITY", i) for i in self.range_m])
        else:
            #This is because x(0,0) is assigned
            self.op.linear_constraints.add(
                lin_expr=[
                    SparsePair(
                        ind=[
                            nm("x", 0, j) for j in self.range_n if j != 0
                        ]
                        ,
                        val=self.w[1:])
                ],
                senses="L",
                rhs=[self.C-self.w[0]],
                names=[nm("CAPACITY_l", 0)])
            # Bins from 1 to l
            self.op.linear_constraints.add(
                lin_expr=[
                    SparsePair(
                        ind=[
                            nm("x", i, j) for j in self.range_n if j != 0
                        ]
                        ,
                        val=self.w[1:]) for i in range(1, self.l)
                ],
                senses="L" * (self.l-1),
                rhs=[self.C] * (self.l-1),
                names=[nm("CAPACITY_l", i) for i in range(1, self.l)])
            self.op.linear_constraints.add(
                lin_expr=[
                    SparsePair(
                        ind=[
                            nm("x", i, j) for j in self.range_n if j != 0
                        ] +[nm("y", i)]
                        ,
                        val=self.w[1:] + [-self.C])
                    for i in self.range_y_vars
                ],
                senses="L" * (self.m - self.l),
                rhs=[0.0] * (self.m - self.l),
                names=[nm("CAPACITY", i) for i in self.range_y_vars])

    def create_cons(self, save_vars=False):
        """Method to populate the constraints"""
        self.create_cons_eq(save_vars)
        self.create_cons_ineq(save_vars)

    def create_obj(self):
        self.op.objective.set_linear([(nm("y", i), 1.0) for i in self.range_y_vars])

    def run_op(self, save_vars=False):
        """Main method, which populates and solve the mathematical model via Python Cplex API
        """

        self.op = OptimizationProblem()

        self.create_params(save_vars=save_vars)
        self.create_vars()
        self.create_obj()
        self.create_cons(save_vars=save_vars)

        # Save the model
        create_folder(self.lp_folder)

        self.op.write(self.lp_folder + "bpp.lp")

        params = ADMMParameters(max_iter=1)
        solver = ADMMOptimizer(params)
        solution = solver.solve(self.op)

        return solution


def toy_op():
    n_items, n_bins, C, w = get_instance_params()
    pb = Bpp(n_items, n_bins, C, w)
    out = pb.run_op()


if __name__ == '__main__':
    # toy_cplex_api()
    toy_op()


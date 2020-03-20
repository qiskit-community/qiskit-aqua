from cplex import SparsePair
from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from qiskit.optimization.algorithms.admm_optimizer import ADMMOptimizer


def create_simple_admm_problem():
    op = OptimizationProblem()
    op.variables.add(names=["x0_1"], types=["B"])
    op.variables.add(names=["x0_2"], types=["B"])
    op.variables.add(names=["u_1"], lb=[0])
    op.variables.add(names=["u_2"], lb=[0])

    # a list with length equal to the number of variables
    # list of SparsePairs
    q_terms = [SparsePair(ind=(0, 1, 2, 3), val=[11, 12, 0, 0]),
               SparsePair(ind=(0, 1, 2, 3), val=[12, 22, 0, 0]),
               SparsePair(ind=(0, 1, 2, 3), val=[0, 0, 33, 34]),
               SparsePair(ind=(0, 1, 2, 3), val=[0, 0, 34, 44])]
    op.objective.set_quadratic(q_terms)

    op.objective.set_linear("x0_1", 1)
    op.objective.set_linear("x0_2", 2)

    op.objective.set_linear("u_1", 3)
    op.objective.set_linear("u_2", 4)

    # A0, vector b0
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1], val=[1, 11])], senses="E", rhs=[11.1],
                              names=["constraint1"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1], val=[2, 22])], senses="E", rhs=[22.22],
                              names=["constraint2"])

    # matrix A1, vector b1
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1], val=[3, 33])], senses="L", rhs=[33.33],
                              names=["constraint3"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1], val=[4, 44])], senses="L", rhs=[44.44],
                              names=["constraint4"])

    # matrix A2, matrix A3
    # original one
    # op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1, 2, 3], val=[5, 55, 555, 5555])], senses="L", rhs=[55.55],
    #                           names=["constraint5"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 2, 1, 3], val=[5, 555, 55, 5555])], senses="L", rhs=[55.55],
                              names=["constraint5"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[0, 1, 2, 3], val=[6, 66, 666, 6666])], senses="L", rhs=[66.66],
                              names=["constraint6"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[2, 3, 0], val=[777, 7777, 7])], senses="L", rhs=[77.77],
                              names=["constraint7"])

    # matrix A4, vector b3
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[2, 3], val=[777, 7777])], senses="L", rhs=[77.77],
                              names=["constraint8"])
    op.linear_constraints.add(lin_expr=[SparsePair(ind=[2, 3], val=[888, 8888])], senses="L", rhs=[88.88],
                              names=["constraint9"])

    op.write("admm-sample.lp")

    return op


if __name__ == '__main__':
    op = create_simple_admm_problem()
    solver = ADMMOptimizer()
    compatible = solver.is_compatible(op)
    print("Problem is compatible: {}".format(compatible))
    solution = solver.solve(op)

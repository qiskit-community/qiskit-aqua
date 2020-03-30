"""
Created: 2020-03-27
@author Claudio Gambella [claudio.gambella1@ie.ibm.com]
"""

from docplex.mp.model import Model

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization import OptimizationProblem
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer


mdl = Model('ex5')

v = mdl.binary_var(name='v')
w = mdl.binary_var(name='w')
t = mdl.binary_var(name='t')

mdl.minimize(v + w + t)
mdl.add_constraint(2 * v + 2 * w + t <= 3, "cons1")
mdl.add_constraint(v + w + t >= 1, "cons2")
mdl.add_constraint(v + w == 1, "cons3")


op = OptimizationProblem()
op.from_docplex(mdl)

# qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
qubo_optimizer = CplexOptimizer()

continuous_optimizer = CplexOptimizer()

admm_params = ADMMParameters(
                             rho_initial=1001, beta=1000, factor_c=900,
                             max_iter=100,  three_block=True,
                             )

solver = ADMMOptimizer(params=admm_params, qubo_optimizer=qubo_optimizer,
                             continuous_optimizer=continuous_optimizer,)
solution = solver.solve(op)

print("results")
print("x={}".format(solution.x))
print("fval={}".format(solution.fval))
print("merits={}".format(solution.results.merits))
print("residuals={}".format(solution.results.residuals))
print("number of iters={}".format(len(solution.results.merits)))

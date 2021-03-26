# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Automatically generate Ising Hamiltonians from general models of optimization problems.
This program converts general models of optimization problems into Ising Hamiltonian.
To write models of optimization problems, DOcplex (Python library for optimization problems)
is used in the program.
(https://cdn.rawgit.com/IBMDecisionOptimization/docplex-doc/master/docs/index.html)

It supports models that consist of the following elements now.

- Binary variables.
- Linear or quadratic object function.
- Equality constraints.

    - Symbols in constraints have to be equal (==).
    - Inequality constraints (e.g. x+y <= 5) are not allowed.

The following is an example of use.

.. code-block:: python

    # Create an instance of a model and variables with DOcplex.
    mdl = Model(name='tsp')
    x = {(i,p): mdl.binary_var(name='x_{0}_{1}'.format(i,p)) for i in range(num_node)
               for p in range(num_node)}

    # Object function
    tsp_func = mdl.sum(ins.w[i,j] * x[(i,p)] * x[(j,(p+1)%num_node)] for i in range(num_node)
                            for j in range(num_node) for p in range(num_node))
    mdl.minimize(tsp_func)

    # Constraints
    for i in range(num_node):
        mdl.add_constraint(mdl.sum(x[(i,p)] for p in range(num_node)) == 1)
    for p in range(num_node):
        mdl.add_constraint(mdl.sum(x[(i,p)] for i in range(num_node)) == 1)

    # Call the method to convert the model into Ising Hamiltonian.
    qubitOp, offset = get_operator(mdl)

    # Calculate with the generated Ising Hamiltonian.
    ee = NumPyMinimumEigensolver(qubitOp)
    result = ee.run()
    print('get_operator')
    print('tsp objective:', result['energy'] + offset)

"""

from typing import Tuple, Union, List
import logging
from math import fsum

import numpy as np
from docplex.mp.constants import ComparisonType
from docplex.mp.constr import LinearConstraint, QuadraticConstraint

try:
    # new location since docplex 2.16.196
    from docplex.mp.dvar import Var
except ImportError:
    # old location until docplex 2.15.194
    from docplex.mp.linear import Var
from docplex.mp.model import Model
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)


def get_operator(mdl: Model, auto_penalty: bool = True,
                 default_penalty: float = 1e5) -> Tuple[WeightedPauliOperator, float]:
    """Generate Ising Hamiltonian from a model of DOcplex.

    Args:
        mdl: A model of DOcplex for a optimization problem.
        auto_penalty: If true, the penalty coefficient is automatically defined
                             by "_auto_define_penalty()".
        default_penalty: The default value of the penalty coefficient for the constraints.
            This value is used if "auto_penalty" is False.

    Returns:
        Operator for the Hamiltonian and a constant shift for the obj function.
    """

    _validate_input_model(mdl)

    # set the penalty coefficient by _auto_define_penalty() or manually.
    if auto_penalty:
        penalty = _auto_define_penalty(mdl, default_penalty)
    else:
        penalty = default_penalty

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sign = 1
    if mdl.is_maximized():
        sign = -1

    # assign variables of the model to qubits.
    q_d = {}
    index = 0
    for i in mdl.iter_variables():
        if i in q_d:
            continue
        q_d[i] = index
        index += 1

    # initialize Hamiltonian.
    num_nodes = len(q_d)
    pauli_list = []
    shift = 0.
    zero = np.zeros(num_nodes, dtype=bool)

    # convert a constant part of the object function into Hamiltonian.
    shift += mdl.get_objective_expr().get_constant() * sign

    # convert linear parts of the object function into Hamiltonian.
    l_itr = mdl.get_objective_expr().iter_terms()
    for j in l_itr:
        z_p = np.zeros(num_nodes, dtype=bool)
        index = q_d[j[0]]
        weight = j[1] * sign / 2
        z_p[index] = True

        pauli_list.append([-weight, Pauli((z_p, zero))])
        shift += weight

    # convert quadratic parts of the object function into Hamiltonian.
    q_itr = mdl.get_objective_expr().iter_quads()
    for i in q_itr:
        index1 = q_d[i[0][0]]
        index2 = q_d[i[0][1]]
        weight = i[1] * sign / 4

        if index1 == index2:
            shift += weight
        else:
            z_p = np.zeros(num_nodes, dtype=bool)
            z_p[index1] = True
            z_p[index2] = True
            pauli_list.append([weight, Pauli((z_p, zero))])

        z_p = np.zeros(num_nodes, dtype=bool)
        z_p[index1] = True
        pauli_list.append([-weight, Pauli((z_p, zero))])

        z_p = np.zeros(num_nodes, dtype=bool)
        z_p[index2] = True
        pauli_list.append([-weight, Pauli((z_p, zero))])

        shift += weight

    # convert constraints into penalty terms.
    for constraint in mdl.iter_constraints():
        right_cst = constraint.get_right_expr().get_constant()
        left_cst = constraint.get_left_expr().get_constant()
        constant = float(right_cst - left_cst)

        # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
        shift += penalty * constant ** 2

        # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
        for __l in _iter_net_linear_coeffs(constraint):
            z_p = np.zeros(num_nodes, dtype=bool)
            index = q_d[__l[0]]
            weight = __l[1]
            z_p[index] = True

            pauli_list.append([penalty * constant * weight, Pauli((z_p, zero))])
            shift += -penalty * constant * weight

        # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
        for __l in _iter_net_linear_coeffs(constraint):
            for l_2 in _iter_net_linear_coeffs(constraint):
                index1 = q_d[__l[0]]
                index2 = q_d[l_2[0]]
                weight1 = __l[1]
                weight2 = l_2[1]
                penalty_weight1_weight2 = penalty * weight1 * weight2 / 4

                if index1 == index2:
                    shift += penalty_weight1_weight2
                else:
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[index1] = True
                    z_p[index2] = True
                    pauli_list.append([penalty_weight1_weight2, Pauli((z_p, zero))])

                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[index1] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli((z_p, zero))])

                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[index2] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli((z_p, zero))])

                shift += penalty_weight1_weight2

    # Remove paulis whose coefficients are zeros.
    qubit_op = WeightedPauliOperator(paulis=pauli_list)

    return qubit_op, shift


def _iter_net_linear_coeffs(constraint: Union[LinearConstraint, QuadraticConstraint]) \
        -> List[Tuple[Var, float]]:
    """
    Builds a list of tuples, where each tuple contains a decision variable and a
    corresponding coefficient of this variable in the constraint.

    Args:
        constraint: A constraint to analyze.

    Returns:
        A list of tuples of variables and coefficients.
    """
    left_expr = constraint.get_left_expr()
    right_expr = constraint.get_right_expr()
    # for linear constraints we may get an instance of Var instead of expression,
    # e.g. x + y = z
    if isinstance(left_expr, Var):
        left_expr = left_expr + 0
    if isinstance(right_expr, Var):
        right_expr = right_expr + 0

    variables = {}
    for var in left_expr.iter_variables():
        variables[var] = left_expr.get_coef(var)
    for var in right_expr.iter_variables():
        variables[var] = variables.get(var, 0.0) - right_expr.get_coef(var)
    return list(variables.items())


def _validate_input_model(mdl: Model) -> None:
    """Check whether an input model is valid. If not, raise an AquaError.

    Args:
         mdl: A model of DOcplex for a optimization problem.

    Raises:
        AquaError: Unsupported input model.
    """
    valid = True

    # validate an object type of the input.
    if not isinstance(mdl, Model):
        raise AquaError('An input model must be docplex.mp.model.Model.')

    # raise an error if the type of the variable is not a binary type.
    for var in mdl.iter_variables():
        if not var.is_binary():
            logger.warning('The type of Variable %s is %s. It must be a binary variable. ',
                           var, var.vartype.short_name)
            valid = False

    # raise an error if the constraint type is not an equality constraint.
    for constraint in mdl.iter_constraints():
        if not constraint.sense == ComparisonType.EQ:
            logger.warning('Constraint %s is not an equality constraint.', constraint)
            valid = False

    if not valid:
        raise AquaError('The input model has unsupported elements.')


def _auto_define_penalty(mdl: Model, default_penalty: float = 1e5) -> float:
    """Automatically define the penalty coefficient.

    This returns object function's (upper bound - lower bound + 1).

    Args:
        mdl: A model of DOcplex for a optimization problem.
        default_penalty: The default value of the penalty coefficient for the constraints.

    Returns:
        The penalty coefficient for the Hamiltonian.
    """

    # if a constraint has float coefficient, return 1e5 for the penalty coefficient.
    terms = []
    for constraint in mdl.iter_constraints():
        terms.append(constraint.cplex_num_rhs())
        terms.extend(term[1] for term in constraint.iter_net_linear_coefs())
    if any(isinstance(term, float) and not term.is_integer() for term in terms):
        logger.warning('Using %f for the penalty coefficient because a float coefficient exists '
                       'in constraints. \nThe value could be too small. '
                       'If so, set the penalty coefficient manually.', default_penalty)
        return default_penalty

    # (upper bound - lower bound) can be calculate as the sum of absolute value of coefficients
    # Firstly, add 1 to guarantee that infeasible answers will be greater than upper bound.
    penalties = [1]
    # add linear terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_terms())
    # add quadratic terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_quads())

    return fsum(penalties)

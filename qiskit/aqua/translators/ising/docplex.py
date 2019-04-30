# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Automatically generate Ising Hamiltonians from general models of optimization problems.
This program converts general models of optimization problems into Ising Hamiltonian.
To write models of optimization problems, DOcplex (Python library for optimization problems) is used in the program.
(https://cdn.rawgit.com/IBMDecisionOptimization/docplex-doc/master/docs/index.html)

It supports models that consist of the following elements now.
- Binary variables.
- Linear or quadratic object function.
- Equality constraints.
  * Symbols in constrains have to be equal (==). Inequality constrains (e.g. x+y <= 5) are not allowed.


The following is an example of use.
---
# Create an instance of a model and variables with DOcplex.
mdl = Model(name='tsp')
x = {(i,p): mdl.binary_var(name='x_{0}_{1}'.format(i,p)) for i in range(num_node) for p in range(num_node)}

# Object function
tsp_func = mdl.sum(ins.w[i,j] * x[(i,p)] * x[(j,(p+1)%num_node)] for i in range(num_node) for j in range(num_node) for p in range(num_node))
mdl.minimize(tsp_func)

# Constrains
for i in range(num_node):
    mdl.add_constraint(mdl.sum(x[(i,p)] for p in range(num_node)) == 1)
for p in range(num_node):
    mdl.add_constraint(mdl.sum(x[(i,p)] for i in range(num_node)) == 1)

# Call the method to convert the model into Ising Hamiltonian.
qubitOp, offset = get_qubitops(mdl)

# Calculate with the generated Ising Hamiltonian.
ee = ExactEigensolver(qubitOp, k=1)
result = ee.run()
print('get_qubitops')
print('tsp objective:', result['energy'] + offset)
---
"""

import logging
from collections import OrderedDict
from math import fsum

import numpy as np
from docplex.mp.constants import ComparisonType
from docplex.mp.model import Model
from qiskit.quantum_info import Pauli

from qiskit.aqua import Operator, AquaError

logger = logging.getLogger(__name__)


def get_qubitops(mdl, auto_penalty=True, default_penalty=1e5):
    """ Generate Ising Hamiltonian from a model of DOcplex.

    Args:
        mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
        auto_penalty (bool): If true, the penalty coefficient is automatically defined by "_auto_define_penalty()".
        default_penalty (float): The default value of the penalty coefficient for the constraints.
            This value is used if "auto_penalty" is False.

    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.
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
    qd = {}
    index = 0
    for i in mdl.iter_variables():
        if i in qd:
            continue
        qd[i] = index
        index += 1

    # initialize Hamiltonian.
    num_nodes = len(qd)
    pauli_list = []
    shift = 0
    zero = np.zeros(num_nodes, dtype=np.bool)

    # convert linear parts of the object function into Hamiltonian.
    l_itr = mdl.get_objective_expr().iter_terms()
    for j in l_itr:
        zp = np.zeros(num_nodes, dtype=np.bool)
        index = qd[j[0]]
        weight = j[1] * sign / 2
        zp[index] = True

        pauli_list.append([-weight, Pauli(zp, zero)])
        shift += weight

    # convert quadratic parts of the object function into Hamiltonian.
    q_itr = mdl.get_objective_expr().iter_quads()
    for i in q_itr:
        index1 = qd[i[0][0]]
        index2 = qd[i[0][1]]
        weight = i[1] * sign / 4

        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[index1] = True
        zp[index2] = True
        pauli_list.append([weight, Pauli(zp, zero)])

        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[index1] = True
        pauli_list.append([-weight, Pauli(zp, zero)])

        zp = np.zeros(num_nodes, dtype=np.bool)
        zp[index2] = True
        pauli_list.append([-weight, Pauli(zp, zero)])

        shift += weight

    # convert constraints into penalty terms.
    for constraint in mdl.iter_constraints():
        constant = constraint.right_expr.get_constant()

        # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
        shift += penalty * constant ** 2

        # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
        for l in constraint.left_expr.iter_terms():
            zp = np.zeros(num_nodes, dtype=np.bool)
            index = qd[l[0]]
            weight = l[1]
            zp[index] = True

            pauli_list.append([penalty * constant * weight, Pauli(zp, zero)])
            shift += -penalty * constant * weight

        # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
        for l in constraint.left_expr.iter_terms():
            for l2 in constraint.left_expr.iter_terms():
                index1 = qd[l[0]]
                index2 = qd[l2[0]]
                weight1 = l[1]
                weight2 = l2[1]
                penalty_weight1_weight2 = penalty * weight1 * weight2 / 4

                if index1 == index2:
                    shift += penalty_weight1_weight2
                else:
                    zp = np.zeros(num_nodes, dtype=np.bool)
                    zp[index1] = True
                    zp[index2] = True
                    pauli_list.append([penalty_weight1_weight2, Pauli(zp, zero)])

                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[index1] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli(zp, zero)])

                zp = np.zeros(num_nodes, dtype=np.bool)
                zp[index2] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli(zp, zero)])

                shift += penalty_weight1_weight2

    # Remove paulis whose coefficients are zeros.
    qubitOp = Operator(paulis=pauli_list)
    qubitOp.zeros_coeff_elimination()

    return qubitOp, shift


def _validate_input_model(mdl):
    """ Check whether an input model is valid. If not, raise an AquaError

    Args:
         mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
    """
    valid = True

    # validate an object type of the input.
    if not isinstance(mdl, Model):
        raise AquaError('An input model must be docplex.mp.model.Model.')

    # raise an error if the type of the variable is not a binary type.
    for var in mdl.iter_variables():
        if not var.is_binary():
            logger.warning(
                'The type of Variable {} is {}. It must be a binary variable. '.format(var, var.vartype.short_name))
            valid = False

    # raise an error if the constraint type is not an equality constraint.
    for constraint in mdl.iter_constraints():
        if not constraint.sense == ComparisonType.EQ:
            logger.warning('Constraint {} is not an equality constraint.'.format(constraint))
            valid = False

    if not valid:
        raise AquaError('The input model has unsupported elements.')


def _auto_define_penalty(mdl, default_penalty=1e5):
    """ Automatically define the penalty coefficient.
    This returns object function's (upper bound - lower bound + 1).


    Args:
        mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
        default_penalty (float): The default value of the penalty coefficient for the constraints.

    Returns:
        float: The penalty coefficient for the Hamiltonian.
    """

    # if a constraint has float coefficient, return 1e5 for the penalty coefficient.
    terms = []
    for constraint in mdl.iter_constraints():
        terms.append(constraint.right_expr.get_constant())
        terms.extend(term[1] for term in constraint.left_expr.iter_terms())
    if any(isinstance(term, float) and not term.is_integer() for term in terms):
        logger.warning('Using %f for the penalty coefficient because a float coefficient exists in constraints. \n'
                       'The value could be too small. If so, set the penalty coefficient manually.', default_penalty)
        return default_penalty

    # (upper bound - lower bound) can be calculate as the sum of absolute value of coefficients
    # Firstly, add 1 to guarantee that infeasible answers will be greater than upper bound.
    penalties = [1]
    # add linear terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_terms())
    # add quadratic terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_quads())

    return fsum(penalties)


def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.

    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, dict) or isinstance(state_vector, OrderedDict):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x

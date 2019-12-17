# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" ast utilities """

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import And, Or, Not, Xor


def get_ast(var_to_lit_map, clause):
    """ get ast """
    # only a single variable
    if isinstance(clause, Symbol):
        return 'lit', var_to_lit_map[clause.binary_symbols.pop()]
    # only a single negated variable
    elif isinstance(clause, Not):
        return 'lit', var_to_lit_map[clause.binary_symbols.pop()] * -1
    # only a single clause
    elif isinstance(clause, (And, Or, Xor)):
        return (str(type(clause)).lower(), *[get_ast(var_to_lit_map, v) for v in clause.args])

    return None

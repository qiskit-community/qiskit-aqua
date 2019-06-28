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


def normalize_literal_indices(raw_ast, raw_indices):
    idx_mapping = {r: i + 1 for r, i in zip(sorted(raw_indices), range(len(raw_indices)))}
    if raw_ast[0] == 'and' or raw_ast[0] == 'or':
        clauses = []
        for c in raw_ast[1:]:
            if c[0] == 'lit':
                clauses.append(('lit', (idx_mapping[c[1]]) if c[1] > 0 else (-idx_mapping[-c[1]])))
            elif (c[0] == 'or' or c[0] == 'and') and (raw_ast[0] != c[0]):
                clause = []
                for l in c[1:]:
                    clause.append(('lit', (idx_mapping[l[1]]) if l[1] > 0 else (-idx_mapping[-l[1]])))
                clauses.append((c[0], *clause))
            else:
                raise AquaError('Unrecognized logical expression: {}'.format(raw_ast))
    elif raw_ast[0] == 'const' or raw_ast[0] == 'lit':
        return raw_ast
    else:
        raise AquaError('Unrecognized root expression type: {}.'.format(raw_ast[0]))
    return (raw_ast[0], *clauses)


def get_ast(var_to_lit_map, clause):
    from sympy.core.symbol import Symbol
    from sympy.logic.boolalg import And, Or, Not, Xor
    # only a single variable
    if isinstance(clause, Symbol):
        return 'lit', var_to_lit_map[clause.binary_symbols.pop()]
    # only a single negated variable
    elif isinstance(clause, Not):
        return 'lit', var_to_lit_map[clause.binary_symbols.pop()] * -1
    # only a single clause
    elif isinstance(clause, Or) or isinstance(clause, And) or isinstance(clause, Xor):
        return (str(type(clause)).lower(), *[get_ast(var_to_lit_map, v) for v in clause.args])

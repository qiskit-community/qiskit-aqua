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

""" classical algorithms packages """

from .exact_eigen_solver.exact_eigen_solver import ExactEigensolver
from .exact_ls_solver.exact_ls_solver import ExactLSsolver
from .svm.svm_classical import SVM_Classical

__all__ = ['ExactEigensolver',
           'ExactLSsolver',
           'SVM_Classical']
try:
    from cplex import Cplex
    from .cplex.cplex_ising import CPLEX_Ising
    __all__ += ['CPLEX_Ising']
except ImportError:
    pass

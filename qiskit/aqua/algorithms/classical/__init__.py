# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" classical algorithms packages """

from .classical_algorithm import ClassicalAlgorithm
from .exact_eigen_solver.exact_eigen_solver import ExactEigensolver
from .exact_ls_solver.exact_ls_solver import ExactLSsolver
from .svm.svm_classical import SVM_Classical
from .cplex.cplex_ising import CPLEX_Ising

__all__ = ['ClassicalAlgorithm',
           'ExactEigensolver',
           'ExactLSsolver',
           'SVM_Classical',
           'CPLEX_Ising']

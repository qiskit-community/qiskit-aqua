# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from .exacteigensolver.exacteigensolver import ExactEigensolver
from .svm.svm_classical import SVM_Classical

__all__ = ['ExactEigensolver',
           'SVM_Classical']

try:
    from .cplex.cplex_ising import CPLEX_Ising
    __all__ += ['CPLEX_Ising']
except ImportError:
    raise ImportWarning('CPLEX is not installed. See https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html')

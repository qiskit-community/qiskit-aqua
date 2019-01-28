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

from abc import ABC, abstractmethod


class _QSVM_Kernel_ABC(ABC):
    """Abstract base class for the binary classifier and the multiclass classifier."""

    def __init__(self, qalgo):

        self._qalgo = qalgo
        self._ret = {}

    @abstractmethod
    def run(self):
        raise NotImplementedError("Must have implemented this.")

    @property
    def ret(self):
        return self._ret

    @ret.setter
    def ret(self, new_ret):
        self._ret = new_ret

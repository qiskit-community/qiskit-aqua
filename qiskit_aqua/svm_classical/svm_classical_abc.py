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

class SVM_Classical_ABC(ABC):
    """
    abstract base class for the binary classifier and the multiclass classifier
    """
    def init_args(self, training_dataset, test_dataset, datapoints, print_info, multiclass_alg):
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = list(self.training_dataset.keys())
        self.print_info = print_info
        self.multiclass_alg = multiclass_alg

    @abstractmethod
    def run(self):
        raise NotImplementedError( "Should have implemented this" )



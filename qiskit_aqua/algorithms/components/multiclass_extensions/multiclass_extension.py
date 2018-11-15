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


class MulticlassExtension(ABC):
    """
        Base class for multiclass extension.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self, configuration=None):
        self._configuration = configuration

    @property
    def configuration(self):
        """Return configuration"""
        return self._configuration

    def init_params(self, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        self.init_args(**args)

    def init_args(self, estimator_cls, code_size=None, params=None):
        self.estimator_cls = estimator_cls
        self.code_size = code_size
        self.params = params

    @abstractmethod
    def train(self, x, y):
        """
        training multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def test(self, x, y):
        """
        testing multiple estimators each for distinguishing a pair of classes.
        Args:
            x (numpy.ndarray): input points
            y (numpy.ndarray): input labels
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        """
        applying multiple estimators for prediction
        Args:
            x (numpy.ndarray): input points
        """
        raise NotImplementedError()

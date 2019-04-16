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
import copy
from qiskit.aqua.parser import JSONSchema
from qiskit.aqua import AquaError
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QiskitFinanceError(AquaError): None

class DataType(Enum):
    DAILYADJUSTED = 'Daily (adj)'
    DAILY = 'Daily'
    BID = 'Bid'
    ASK = 'Ask'
    
class BaseDataProvider(ABC):
    """
    This module implements the abstract base class for data_provider modules
    within Qiskit Finance. 

    To create add-on data_provider module subclass the BaseDataProvider class in this module.
    Doing so requires that the required driver interface is implemented.

    To use the subclasses, please see
    https://github.com/Qiskit/qiskit-tutorials/qiskit/finance/data_providers/time_series.ipynb

    """

    @abstractmethod
    def __init__(self):
        self.check_driver_valid()
        self._configuration = copy.deepcopy(self.CONFIGURATION)

    @property
    def configuration(self):
        """Return driver configuration."""
        return self._configuration

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        pass

    @staticmethod
    def check_driver_valid():
        """Checks if driver is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        jsonSchema = JSONSchema(schema_dict)
        schema_property_names = jsonSchema.get_default_section_names()
        json_dict = {}
        for property_name in schema_property_names:
            if property_name in args_dict:
                json_dict[property_name] = args_dict[property_name]

        jsonSchema.validate(json_dict)

    @abstractmethod
    def run(self):
        pass

    # gets coordinates suitable for plotting
    # it does not have to be overridden in non-abstract derived classes.
    def get_coordinates(self):
        import numpy as np
        # Coordinates for visualisation purposes
        xc = np.zeros([self._n, 1])
        yc = np.zeros([self._n, 1])
        xc = (np.random.rand(self._n) - 0.5) * 1
        yc = (np.random.rand(self._n) - 0.5) * 1
        #for (cnt, s) in enumerate(self.tickers):
        #xc[cnt, 1] = self.data[cnt][0]
        # yc[cnt, 0] = self.data[cnt][-1]
        return xc, yc

    # it does not have to be overridden in non-abstract derived classes.
    def get_covariance(self):
        if not self._data: return None   
        import numpy as np
        if not self._data: return None
        self.cov = np.cov(self._data, rowvar = True)
        return self.cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_similarity_matrix(self):
        if not self._data: return None    
        import numpy as np
        try:
          import fastdtw
          self.rho = np.zeros((self._n, self._n))
          for ii in range(0, self._n):
            self.rho[ii,ii] = 1.
            for jj in range(ii + 1, self._n):
                thisRho, path = fastdtw.fastdtw(self._data[ii], self._data[jj])
                thisRho = 1.0 / thisRho
                self.rho[ii, jj] = thisRho
                self.rho[jj, ii] = thisRho
        except ImportError:
          print("This requires fastdtw package.")
        return self.rho

    # it does not have to be overridden in non-abstract derived classes.
    def plot(self):  
        import matplotlib.pyplot as plt
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        print("Evolution of the stock price:")
        for (cnt, s) in enumerate(self._tickers):
            print(s)
            print(self._data[cnt])
            plt.plot(self._data[cnt], label=s)
        plt.legend()
        plt.xticks(rotation=90)
        #plt.title("Evolution of the adjusted closing price")
        plt.show()
        if self._n <= 1: 
            print("Not enough data to plot covariance or time-series similarity. Please use at least two tickers.")
            return
        self.get_similarity_matrix()
        print("A time-series similarity measure:")
        print(self.rho)
        plt.subplot(211)
        plt.imshow(self.rho)
        self.get_covariance()
        print("A covariance matrix:")
        print(self.cov)
        plt.subplot(212)
        plt.imshow(self.cov)
        plt.show()     
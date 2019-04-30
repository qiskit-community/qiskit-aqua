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

from abc import ABC, abstractmethod
import logging
import copy
from enum import Enum

import numpy as np
import fastdtw

from qiskit.aqua import AquaError
from qiskit.aqua.parser import JSONSchema

logger = logging.getLogger(__name__)


class QiskitFinanceError(AquaError):
    pass


# Note: Not all DataProviders support all stock markets.
# Check the DataProvider before use.
class StockMarket(Enum):
    NASDAQ = 'NASDAQ'
    NYSE = 'NYSE'
    LONDON = 'XLON'
    EURONEXT = 'XPAR'
    SINGAPORE = 'XSES'
    RANDOM = 'RANDOM'


# Note: Not all DataProviders support all data types.
# Check the DataProvider before use.
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
        Initialize via section dictionary. N.B. Not in use at the moment.

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
        """ Validates the configuration against the input schema. N.B. Not in use at the moment. """

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
        """ Loads data. """
        pass

    # it does not have to be overridden in non-abstract derived classes.
    def get_mean_vector(self):
        """ Returns a vector containing the mean value of each asset. 
        
    Returns:
        mean (numpy.ndarray) : a per-asset mean vector.        
        """
        try:
            if not self._data:
                raise QiskitFinanceError(
                    'No data loaded, yet. Please run the method run() first to load the data.'
                )
        except AttributeError:
            raise QiskitFinanceError(
                'No data loaded, yet. Please run the method run() first to load the data.'
            )
        self.mean = np.mean(self._data, axis=1)
        return self.mean

    # it does not have to be overridden in non-abstract derived classes.
    def get_period_return_mean_vector(self):
        """ Returns a vector containing the mean value of each asset.

    Returns:
        mean (numpy.ndarray) : a per-asset mean vector.
        """
        try:
            if not self._data:
                raise QiskitFinanceError(
                    'No data loaded, yet. Please run the method run() first to load the data.'
                )
        except AttributeError:
            raise QiskitFinanceError(
                'No data loaded, yet. Please run the method run() first to load the data.'
            )

        period_returns = np.array(self._data)[:, 1:] / np.array(self._data)[:, :-1] - 1

        self.period_return_mean = np.mean(period_returns, axis=1)
        return self.period_return_mean

    # it does not have to be overridden in non-abstract derived classes.
    def get_covariance_matrix(self):
        """ Returns the covariance matrix. 
        
    Returns:
        rho (numpy.ndarray) : an asset-to-asset covariance matrix.        
        """
        try:
            if not self._data:
                raise QiskitFinanceError(
                    'No data loaded, yet. Please run the method run() first to load the data.'
                )
        except AttributeError:
            raise QiskitFinanceError(
                'No data loaded, yet. Please run the method run() first to load the data.'
            )
        self.cov = np.cov(self._data, rowvar=True)
        return self.cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_period_return_covariance_matrix(self):
        """ Returns a vector containing the mean value of each asset.

    Returns:
        mean (numpy.ndarray) : a per-asset mean vector.
        """
        try:
            if not self._data:
                raise QiskitFinanceError(
                    'No data loaded, yet. Please run the method run() first to load the data.'
                )
        except AttributeError:
            raise QiskitFinanceError(
                'No data loaded, yet. Please run the method run() first to load the data.'
            )

        period_returns = np.array(self._data)[:, 1:] / np.array(self._data)[:, :-1] - 1

        self.period_return_cov = np.cov(period_returns)
        return self.period_return_cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_similarity_matrix(self):
        """ Returns time-series similarity matrix computed using dynamic time warping. 

    Returns:
        rho (numpy.ndarray) : an asset-to-asset similarity matrix.
        """
        try:
            if not self._data:
                raise QiskitFinanceError(
                    'No data loaded, yet. Please run the method run() first to load the data.'
                )
        except AttributeError:
            raise QiskitFinanceError(
                'No data loaded, yet. Please run the method run() first to load the data.'
            )
        self.rho = np.zeros((self._n, self._n))
        for ii in range(0, self._n):
            self.rho[ii, ii] = 1.
            for jj in range(ii + 1, self._n):
                thisRho, path = fastdtw.fastdtw(self._data[ii], self._data[jj])
                thisRho = 1.0 / thisRho
                self.rho[ii, jj] = thisRho
                self.rho[jj, ii] = thisRho
        return self.rho

    # gets coordinates suitable for plotting
    # it does not have to be overridden in non-abstract derived classes.
    def get_coordinates(self):
        """ Returns random coordinates for visualisation purposes. """
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

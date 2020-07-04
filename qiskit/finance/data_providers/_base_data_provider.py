# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module implements the abstract base class for data_provider modules the finance module."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import logging
from enum import Enum

import numpy as np
import fastdtw

from qiskit.aqua import aqua_globals
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class StockMarket(Enum):
    """ Stock Market enum """
    LONDON = 'XLON'
    EURONEXT = 'XPAR'
    SINGAPORE = 'XSES'


class BaseDataProvider(ABC):
    """The abstract base class for data_provider modules within Qiskit's finance module.

    To create add-on data_provider module subclass the BaseDataProvider class in this module.
    Doing so requires that the required driver interface is implemented.

    To use the subclasses, please see
    https://github.com/Qiskit/qiskit-tutorials/blob/master/legacy_tutorials/aqua/finance/data_providers/time_series.ipynb

    """

    @abstractmethod
    def __init__(self) -> None:
        self._data = None  # type: Optional[List]
        self._n = 0  # pylint: disable=invalid-name
        self.period_return_mean = None
        self.cov = None
        self.period_return_cov = None
        self.rho = None  # type: Optional[np.ndarray]
        self.mean = None

    @abstractmethod
    def run(self) -> None:
        """ Loads data. """
        pass

    # it does not have to be overridden in non-abstract derived classes.
    def get_mean_vector(self) -> np.ndarray:
        """ Returns a vector containing the mean value of each asset.

        Returns:
            a per-asset mean vector.
        Raises:
            QiskitFinanceError: no data loaded
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

    @staticmethod
    def _divide(val_1, val_2):
        if val_2 == 0:
            if val_1 == 0:
                return 1
            logger.warning('Division by 0 on values %f and %f', val_1, val_2)
            return np.nan
        return val_1 / val_2

    # it does not have to be overridden in non-abstract derived classes.
    def get_period_return_mean_vector(self) -> np.ndarray:
        """
        Returns a vector containing the mean value of each asset.

        Returns:
            a per-asset mean vector.
        Raises:
            QiskitFinanceError: no data loaded
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
        _div_func = np.vectorize(BaseDataProvider._divide)
        period_returns = _div_func(np.array(self._data)[:, 1:], np.array(self._data)[:, :-1]) - 1
        self.period_return_mean = np.mean(period_returns, axis=1)
        return self.period_return_mean

    # it does not have to be overridden in non-abstract derived classes.
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Returns the covariance matrix.

        Returns:
            an asset-to-asset covariance matrix.
        Raises:
            QiskitFinanceError: no data loaded
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
    def get_period_return_covariance_matrix(self) -> np.ndarray:
        """
        Returns a vector containing the mean value of each asset.

        Returns:
            a per-asset mean vector.
        Raises:
            QiskitFinanceError: no data loaded
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
        _div_func = np.vectorize(BaseDataProvider._divide)
        period_returns = _div_func(np.array(self._data)[:, 1:], np.array(self._data)[:, :-1]) - 1
        self.period_return_cov = np.cov(period_returns)
        return self.period_return_cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Returns time-series similarity matrix computed using dynamic time warping.

        Returns:
            an asset-to-asset similarity matrix.
        Raises:
            QiskitFinanceError: no data loaded
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
        for i_i in range(0, self._n):
            self.rho[i_i, i_i] = 1.
            for j_j in range(i_i + 1, self._n):
                this_rho, _ = fastdtw.fastdtw(self._data[i_i], self._data[j_j])
                this_rho = 1.0 / this_rho
                self.rho[i_i, j_j] = this_rho
                self.rho[j_j, i_i] = this_rho
        return self.rho

    # gets coordinates suitable for plotting
    # it does not have to be overridden in non-abstract derived classes.
    def get_coordinates(self) -> Tuple[float, float]:
        """ Returns random coordinates for visualisation purposes. """
        # Coordinates for visualisation purposes
        x_c = np.zeros([self._n, 1])
        y_c = np.zeros([self._n, 1])
        x_c = (aqua_globals.random.random(self._n) - 0.5) * 1
        y_c = (aqua_globals.random.random(self._n) - 0.5) * 1
        # for (cnt, s) in enumerate(self.tickers):
        # x_c[cnt, 1] = self.data[cnt][0]
        # y_c[cnt, 0] = self.data[cnt][-1]
        return x_c, y_c

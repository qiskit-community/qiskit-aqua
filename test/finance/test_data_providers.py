# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Data Providers """

import unittest
import os
import datetime
from test.finance import QiskitFinanceTestCase
import warnings
import numpy as np
from qiskit.aqua import MissingOptionalLibraryError
from qiskit.finance import QiskitFinanceError
from qiskit.finance.data_providers import (RandomDataProvider,
                                           WikipediaDataProvider,
                                           YahooDataProvider,
                                           StockMarket,
                                           DataOnDemandProvider,
                                           ExchangeDataProvider)


# This can be run as python -m unittest test.test_data_providers.TestDataProviders

class TestDataProviders(QiskitFinanceTestCase):
    """Tests data providers for the Portfolio Optimization and Diversification."""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
        self._quandl_token = os.getenv('QUANDL_TOKEN') if os.getenv('QUANDL_TOKEN') else ''
        self._on_demand_token = os.getenv('ON_DEMAND_TOKEN') if os.getenv('ON_DEMAND_TOKEN') else ''

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", message="unclosed", category=ResourceWarning)

    def test_random_wrong_use(self):
        """ Random wrong use test """
        try:
            rnd = RandomDataProvider(seed=1)
            # Now, the .run() method is expected, which does the actual data loading
            # (and can take seconds or minutes,
            # depending on the data volumes, hence not ok in the constructor)
            with self.subTest('test RandomDataProvider get_covariance_matrix'):
                self.assertRaises(QiskitFinanceError, rnd.get_covariance_matrix)
            with self.subTest('test RandomDataProvider get_similarity_matrix'):
                self.assertRaises(QiskitFinanceError, rnd.get_similarity_matrix)
            wiki = WikipediaDataProvider(
                token=self._quandl_token,
                tickers=["GOOG", "AAPL"],
                start=datetime.datetime(2016, 1, 1),
                end=datetime.datetime(2016, 1, 30)
            )
            # Now, the .run() method is expected, which does the actual data loading
            with self.subTest('test WikipediaDataProvider get_covariance_matrix'):
                self.assertRaises(QiskitFinanceError, wiki.get_covariance_matrix)
            with self.subTest('test WikipediaDataProvider get_similarity_matrix'):
                self.assertRaises(QiskitFinanceError, wiki.get_similarity_matrix)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_yahoo_wrong_use(self):
        """ Yahoo! wrong use test """
        try:
            yahoo = YahooDataProvider(
                tickers=["AEO", "ABBY"],
                start=datetime.datetime(2018, 1, 1),
                end=datetime.datetime(2018, 12, 31)
            )
            # Now, the .run() method is expected, which does the actual data loading
            with self.subTest('test YahooDataProvider get_covariance_matrix'):
                self.assertRaises(QiskitFinanceError, yahoo.get_covariance_matrix)
            with self.subTest('test YahooDataProvider get_similarity_matrix'):
                self.assertRaises(QiskitFinanceError, yahoo.get_similarity_matrix)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_random(self):
        """ random test """
        similarity = np.array([[1.00000000e+00, 6.2284804e-04], [6.2284804e-04, 1.00000000e+00]])
        covariance = np.array([[2.08413157, 0.20842107], [0.20842107, 1.99542187]])
        try:
            rnd = RandomDataProvider(seed=1)
            rnd.run()
            with self.subTest('test RandomDataProvider get_covariance_matrix'):
                np.testing.assert_array_almost_equal(rnd.get_covariance_matrix(),
                                                     covariance, decimal=3)
            with self.subTest('test RandomDataProvider get_similarity_matrix'):
                np.testing.assert_array_almost_equal(rnd.get_similarity_matrix(),
                                                     similarity, decimal=3)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_random_divide_0(self):
        """ Random divide by 0 test """
        # This will create data with some 0 values, it should not throw
        # divide by 0 errors
        try:
            seed = 8888
            num_assets = 4
            stocks = [("TICKER%s" % i) for i in range(num_assets)]
            data = RandomDataProvider(tickers=stocks,
                                      start=datetime.datetime(2016, 1, 1),
                                      end=datetime.datetime(2016, 1, 30),
                                      seed=seed)
            data.run()
            mu_value = data.get_period_return_mean_vector()
            sigma_value = data.get_period_return_covariance_matrix()
            with self.subTest('test get_period_return_mean_vector is numpy array'):
                self.assertIsInstance(mu_value, np.ndarray)
            with self.subTest('test get_period_return_covariance_matrix is numpy array'):
                self.assertIsInstance(sigma_value, np.ndarray)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_wikipedia(self):
        """ wikipedia test """
        try:
            wiki = WikipediaDataProvider(
                token=self._quandl_token,
                tickers=["GOOG", "AAPL"],
                start=datetime.datetime(2016, 1, 1),
                end=datetime.datetime(2016, 1, 30)
            )
            wiki.run()
            similarity = np.array([
                [1.00000000e+00, 8.44268222e-05],
                [8.44268222e-05, 1.00000000e+00]
            ])
            covariance = np.array([
                [269.60118129, 25.42252332],
                [25.42252332, 7.86304499]
            ])
            with self.subTest('test WikipediaDataProvider get_covariance_matrix'):
                np.testing.assert_array_almost_equal(wiki.get_covariance_matrix(),
                                                     covariance, decimal=3)
            with self.subTest('test WikipediaDataProvider get_similarity_matrix'):
                np.testing.assert_array_almost_equal(wiki.get_similarity_matrix(),
                                                     similarity, decimal=3)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except QiskitFinanceError as ex:
            self.skipTest("Test of WikipediaDataProvider skipped: {}".format(str(ex)))
            # The trouble for automating testing is that after 50 tries
            # from one IP address within a day
            # Quandl complains about the free usage tier limits:
            # quandl.errors.quandl_error.LimitExceededError: (Status 429) (Quandl Error QELx01)
            # You have exceeded the anonymous user limit of 50 calls per day. To make more calls
            # today, please register for a free Quandl account and then include your API
            # key with your requests.
            # This gets "dressed" as QiskitFinanceError.
            # This also introduces a couple of seconds of a delay.

    def test_nasdaq(self):
        """ nasdaq test """
        try:
            nasdaq = DataOnDemandProvider(
                token=self._on_demand_token,
                tickers=["GOOG", "AAPL"],
                start=datetime.datetime(2016, 1, 1),
                end=datetime.datetime(2016, 1, 2)
            )
            nasdaq.run()
        except QiskitFinanceError as ex:
            self.skipTest("Test of DataOnDemandProvider skipped {}".format(str(ex)))

    def test_exchangedata(self):
        """ exchange data test """
        try:
            lse = ExchangeDataProvider(
                token=self._quandl_token,
                tickers=["AEO", "ABBY"],
                stockmarket=StockMarket.LONDON,
                start=datetime.datetime(2018, 1, 1),
                end=datetime.datetime(2018, 12, 31)
            )
            lse.run()
            similarity = np.array([
                [1.00000000e+00, 8.44268222e-05],
                [8.44268222e-05, 1.00000000e+00]
            ])
            covariance = np.array(
                [[2.693, -18.65],
                 [-18.65, 1304.422]])
            with self.subTest('test ExchangeDataProvider get_covariance_matrix'):
                np.testing.assert_array_almost_equal(lse.get_covariance_matrix(),
                                                     covariance, decimal=3)
            with self.subTest('test ExchangeDataProvider get_similarity_matrix'):
                np.testing.assert_array_almost_equal(lse.get_similarity_matrix(),
                                                     similarity, decimal=3)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except QiskitFinanceError as ex:
            self.skipTest("Test of ExchangeDataProvider skipped {}".format(str(ex)))

    def test_yahoo(self):
        """ Yahoo data test """
        try:
            yahoo = YahooDataProvider(
                tickers=["AEO", "ABBY"],
                start=datetime.datetime(2018, 1, 1),
                end=datetime.datetime(2018, 12, 31)
            )
            yahoo.run()
            similarity = np.array([
                [1.00000000e+00, 8.44268222e-05],
                [8.44268222e-05, 1.00000000e+00]
            ])
            covariance = np.array(
                [[7.174e+00, -1.671e-04],
                 [-1.671e-04, 1.199e-06]])
            with self.subTest('test YahooDataProvider get_covariance_matrix'):
                np.testing.assert_array_almost_equal(yahoo.get_covariance_matrix(),
                                                     covariance, decimal=1)
            with self.subTest('test YahooDataProvider get_similarity_matrix'):
                np.testing.assert_array_almost_equal(yahoo.get_similarity_matrix(),
                                                     similarity, decimal=1)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except QiskitFinanceError as ex:
            self.skipTest("Test of YahooDataProvider skipped {}".format(str(ex)))


if __name__ == '__main__':
    unittest.main()

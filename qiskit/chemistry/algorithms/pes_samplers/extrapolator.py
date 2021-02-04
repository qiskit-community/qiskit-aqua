# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An implementation to extrapolate variational parameters."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, cast

import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA, KernelPCA
from qiskit.aqua import AquaError


class Extrapolator(ABC):
    """
    This class is based on performing extrapolation of parameters of a wavefunction for a
    variational algorithm defined in the variational forms as part of the Qiskit Aqua module.
    This concept is based on fitting a set of (point,parameter) data to some specified
    function and predicting the optimal variational parameters for the next point. This
    technique is aimed towards providing a better starting point for the variational algorithm,
    in addition to bootstrapping techniques, ultimately reducing the total number of function
    evaluations.

    Each instance of an Extrapolator requires a dictionary where each item consist of a point
    (key) and a list of variational parameters (value) for that given point. In practice, a Sampler
    Class can utilize the Extrapolator as a wrapper. The Extrapolator class then extracts optimal
    variational parameters from the previous points for use in extrapolation. For instance, one can
    utilize the Extrapolator to accelerate the computation of the Born-Oppenheimer Potential Energy
    Surface (BOPES) for a given molecule. In this case, each point can represent the interatomic
    distance and the list of parameters represent rotational parameters in a quantum circuit,
    in the context of computing the bond dissociation profile for a diatomic molecule.
    NOTE: However this is not a requirement - once an instance of the Extrapolator class is created,
    extrapolation can proceed by specifying the point(s) of interest and the dictionary of
    (point, parameter) pairs for a problem.

    There are two types of Extrapolators: external/wrapper and internal.
    The external/wrapper extrapolator specifies the number of previous points or data window
    within which to perform the extrapolation as well as the dimensionality/space to
    perform the extrapolation. For instance, one can utilize the PCA Extrapolator as an external
    extrapolator that sets the data window and transforms the variational parameters in PCA space
    before the actual extrapolation is executed. The internal extrapolator can then proceed via
    linear regression/spline fitting of variational parameters to predict a parameter set.
    """

    @abstractmethod
    def extrapolate(self, points: List[float],
                    param_dict: Dict[float, List[float]]) -> Dict[float, List[float]]:
        """
        Abstract method to extrapolate point(s) of interest.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
                and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        raise NotImplementedError()

    @staticmethod
    def factory(mode: str, **kwargs) -> 'Extrapolator':
        """
        Factory method for constructing extrapolators.

        Args:
            mode: Extrapolator to instantiate. Can be one of:
                - 'window'
                - 'poly'
                - 'diff_model'
                - 'pca'
                - 'l1'
            kwargs: arguments to be passed to the constructor of an extrapolator

        Returns:
            A newly created extrapolator instance.

        Raises:
            AquaError: if specified mode is unknown.
        """
        if mode == 'window':
            return WindowExtrapolator(**kwargs)
        elif mode == 'poly':
            return PolynomialExtrapolator(**kwargs)
        elif mode == 'diff_model':
            return DifferentialExtrapolator(**kwargs)
        elif mode == 'pca':
            return PCAExtrapolator(**kwargs)
        elif mode == 'l1':
            return SieveExtrapolator(**kwargs)
        else:
            raise AquaError('No extrapolator called {}'.format(mode))


class PolynomialExtrapolator(Extrapolator):
    """
    An extrapolator based on fitting each parameter to a polynomial function of a user-specified
    degree.

    WARNING: Should only be used with window. Using no window includes points after
    the point being extrapolated in the data window.
    """

    def __init__(self, degree: int = 1) -> None:
        """
        Constructor.

        Args:
            degree: Degree of polynomial to use for fitting in extrapolation.
        """

        self._degree = degree

    def extrapolate(self, points: List[float], param_dict: Optional[Dict[float, List[float]]]) \
            -> Dict[float, List[float]]:
        """
        Extrapolate at specified point of interest given a set of variational parameters.
        Extrapolation is based on a polynomial function/spline fitting with a user-specified
        degree.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
                and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        param_arr = np.transpose(list(param_dict.values()))
        data_points = list(param_dict.keys())

        ret_param_arr = []
        for params in param_arr:
            coefficients = np.polyfit(data_points, params, deg=self._degree)
            poly = np.poly1d(coefficients)
            ret_param_arr += [poly(points)]
        ret_param_arr = np.transpose(ret_param_arr).tolist()
        ret_params = dict(zip(points, ret_param_arr))
        return ret_params


class DifferentialExtrapolator(Extrapolator):
    """
    An extrapolator based on treating each param set as a point in space, and fitting a
    Hamiltonian which evolves each point to the next. The user specifies the type of regression
    model to perform fitting, and a degree which adds derivatives to the values in the point
    vector; serving as features for the regression model.
    WARNING: Should only be used with window. Using no window includes points after the
    point being extrapolated in the data window.
    """

    def __init__(self,
                 degree: int = 1,
                 model: Optional[Union[linear_model.LinearRegression, linear_model.Ridge,
                                       linear_model.RidgeCV, linear_model.SGDRegressor]] = None) \
            -> None:
        """
        Constructor.

        Args:
            model: Regression model (from sklearn) to be used for fitting
                variational parameters. Currently supports the following models:
                LinearRegression(), Ridge(), RidgeCV(), and SGDRegressor().

            degree: Specifies (degree -1) derivatives to be added as
                'features' in regression model.

        """
        self._degree = degree
        self._model = model or linear_model.LinearRegression()

    def extrapolate(self, points: List[float], param_dict: Optional[Dict[float, List[float]]]) \
            -> Dict[float, List[float]]:
        """
        Extrapolate at specified point of interest given a set of variational parameters.
        Each parameter list and list of numerical gradients is treated as a single point
        in vector space. The regression model tries to fit a Hamiltonian that describes
        the evolution from one parameter set (and its gradient features) at point r,
        to another parameter set at point, r + epsilon. The regression model is then
        used to predict the parameter set at the point of interest. Note that this
        extrapolation technique does not explicitly use the spacing of the points
        (step size) but rather infers it from the list of parameter values.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
            and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        response = list(param_dict.values())[1:]
        features = [list(param_dict.values())]
        for i in range(self._degree - 1):
            grad = np.gradient(features[i], axis=0)
            features.append(list(grad))
        features = np.concatenate(features, axis=1)
        self._model.fit(features[:-1], response)
        next_params = np.asarray(self._model.predict([features[-1]])[0].tolist())
        ret_params = {point: next_params for point in points}
        return cast(Dict[float, List[float]], ret_params)


class WindowExtrapolator(Extrapolator):
    """
    An extrapolator which wraps another extrapolator, limiting the internal extrapolator's
    ground truth parameter set to a fixed window size.
    """

    def __init__(self,
                 extrapolator: Union[PolynomialExtrapolator,
                                     DifferentialExtrapolator] = None,
                 window: int = 2) -> None:
        """
        Constructor.

        Args:
            extrapolator: 'internal' extrapolator that performs extrapolation on
                variational parameters based on data window

            window: Number of previous points to use for extrapolation. A value of zero
                indicates that all previous points will be used for bootstrapping.
        """
        self._extrapolator = extrapolator
        self._window = window

    def extrapolate(self, points: List[float], param_dict: Optional[Dict[float, List[float]]]) \
            -> Dict[float, List[float]]:
        """
        Extrapolate at specified point of interest given a set of variational parameters.
        Based on the specified window, a subset of the data points will be used for
        extrapolation. A default window of 2 points is used, while a value of zero indicates
        that all previous points will be used for extrapolation. This method defines the
        data window before performing the internal extrapolation.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
                and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        ret_params = {}
        sorted_points = sorted(points)
        reference_points = [pt for pt in sorted(param_dict.keys()) if pt < max(sorted_points)]

        for bottom_index, bottom in enumerate(reference_points):
            if bottom_index < len(reference_points) - 1:
                top = reference_points[bottom_index + 1]
            else:
                top = float('inf')
            extrapolation_group = [pt for pt in sorted_points if bottom < pt <= top]
            window_points = [pt for pt in reference_points if pt <= bottom]
            if len(window_points) > self._window:
                window_points = window_points[-self._window:]
            window_param_dict = {pt: param_dict[pt] for pt in window_points}
            if extrapolation_group:
                ret_params.update(self._extrapolator.extrapolate(extrapolation_group,
                                                                 param_dict=window_param_dict))
        return ret_params

    @property
    def extrapolator(self) -> Extrapolator:
        """Returns the internal extrapolator.

        Returns:
            The internal extrapolator.
        """
        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, extrapolator: Union[PolynomialExtrapolator,
                                               DifferentialExtrapolator]) -> None:
        """Sets the internal extrapolator.

        Args:
            extrapolator: The internal extrapolator to set.
        """
        self._extrapolator = extrapolator

    @property
    def window(self) -> int:
        """Returns the size of the window.

        Returns:
            The size of the window.
        """
        return self._window

    @window.setter
    def window(self, window: int) -> None:
        """Set the size of the window

        Args:
            window: the size of the window to set.
        """
        self._window = window


class PCAExtrapolator(Extrapolator):
    """
    A wrapper extrapolator which reduces the points' dimensionality with PCA,
    performs extrapolation in the transformed pca space, and inverse transforms the
    results before returning.
    A user specifies the kernel within how the PCA transformation should be done.
    """

    def __init__(self,
                 extrapolator: Optional[Union[PolynomialExtrapolator,
                                              DifferentialExtrapolator]] = None,
                 kernel: Optional[str] = None,
                 window: int = 2) -> None:
        """
        Constructor.

        Args:
            extrapolator: 'internal' extrapolator that performs extrapolation on
                variational parameters based on data window.
            kernel: Kernel (from sklearn) that specifies how dimensionality
                reduction should be done for PCA. Default value is None, and switches
                the extrapolation to standard PCA.
            window: Number of previous points to use for extrapolation.

        Raises:
            AquaError: if kernel is not defined in sklearn module.
        """
        self._extrapolator = WindowExtrapolator(extrapolator=extrapolator, window=window)
        self._kernel = kernel
        if self._kernel is None:
            self._pca_model = PCA()
        elif self._kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            self._pca_model = KernelPCA(kernel=self._kernel, fit_inverse_transform=True)
        else:
            raise AquaError('PCA kernel type {} not found'.format(self._kernel))

    def extrapolate(self, points: List[float], param_dict: Optional[Dict[float, List[float]]]) \
            -> Dict[float, List[float]]:
        """
        Extrapolate at specified point of interest given a set of variational parameters.
        This method transforms the parameters in PCA space before performing the internal
        extrapolation. The parameters are transformed back to regular space after extrapolation.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
            and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        # run pca fitting and extrapolate in pca space
        self._pca_model.fit(list(param_dict.values()))
        updated_params = {pt: self._pca_model.transform([param_dict[pt]])[0]
                          for pt in list(param_dict.keys())}
        output_params = self._extrapolator.extrapolate(points, param_dict=updated_params)

        ret_params = {point: self._pca_model.inverse_transform(param) if not param else []
                      for (point, param) in output_params.items()}
        return ret_params


class SieveExtrapolator(Extrapolator):
    """
    A wrapper extrapolator which clusters the parameter values - either before
    extrapolation, after, or both - into two large and small clusters, and sets the
    small clusters' parameters to zero.
    """

    def __init__(self,
                 extrapolator: Optional[Union[PolynomialExtrapolator,
                                              DifferentialExtrapolator]] = None,
                 window: int = 2,
                 filter_before: bool = True,
                 filter_after: bool = True) -> None:
        """
        Constructor.

        Args:
            extrapolator: 'internal' extrapolator that performs extrapolation on
                variational parameters based on data window.
            window: Number of previous points to use for extrapolation.
            filter_before: Keyword to perform clustering before extrapolation.
            filter_after: Keyword to perform clustering after extrapolation.

        """
        self._extrapolator = WindowExtrapolator(extrapolator=extrapolator, window=window)
        self._filter_before = filter_before
        self._filter_after = filter_after

    def extrapolate(self, points: List[float], param_dict: Optional[Dict[float, List[float]]]) \
            -> Dict[float, List[float]]:
        """
        Extrapolate at specified point of interest given a set of variational parameters.
        Based on the specified window, a subset of the data points will be used for
        extrapolation. A default window of 2 points is used, while a value of zero indicates
        that all previous points will be used for extrapolation. This method finds a cutoff distance
        based on the maximum average distance or 'gap' between the average values of the variational
        parameters. This cutoff distance is used as a criteria to divide the parameters into two
        clusters by setting all parameters that are below the cutoff distance to zero.

        Args:
            points: List of point(s) to be used for extrapolation. Can represent
                some degree of freedom, ex, interatomic distance.
            param_dict: Dictionary of variational parameters. Each key is the point
                and the value is a list of the variational parameters.

        Returns:
            Dictionary of variational parameters for extrapolated point(s).
        """
        # determine clustering cutoff
        param_arr = np.transpose(list(param_dict.values()))
        param_averages = np.array(sorted(np.average(np.log10(np.abs(param_arr)), axis=0)))
        gaps = param_averages[1:] - param_averages[:-1]
        max_gap = int(np.argmax(gaps))
        sieve_cutoff = 10 ** np.average([param_averages[max_gap], param_averages[max_gap + 1]])

        if self._filter_before:
            filtered_dict = {point: list(map(lambda x: x if np.abs(x) > sieve_cutoff else 0, param))
                             for (point, param) in param_dict.items()}
            output_params = self._extrapolator.extrapolate(points, param_dict=filtered_dict)
        else:
            output_params = self._extrapolator.extrapolate(points, param_dict=param_dict)

        if self._filter_after:
            ret_params = \
                cast(Dict[float, List[float]],
                     {point: np.asarray(list(map(lambda x: x
                                                 if np.abs(x) > sieve_cutoff else 0, param)))
                      for (point, param) in output_params.items()})
        else:
            ret_params = cast(Dict[float, List[float]], np.asarray(output_params))
        return ret_params

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

"""Methods for querying the solution to an optimization problem."""

from qiskit.optimization.results.quality_metrics import QualityMetrics
from qiskit.optimization.results.solution_status import SolutionStatus

from qiskit.optimization.utils.base import BaseInterface


class SolutionInterface(BaseInterface):
    """Methods for querying the solution to an optimization problem."""

    # method = SolutionMethod()
    # """See `SolutionMethod()` """
    # quality_metric = QualityMetric()
    # """See `QualityMetric()` """
    status = SolutionStatus()
    """See `SolutionStatus()` """

    def __init__(self):
        """Creates a new SolutionInterface.

        The solution interface is exposed by the top-level `Cplex` class
        as Cplex.solution.  This constructor is not meant to be used
        externally.
        """
        # pylint: disable=useless-super-delegation
        super().__init__()
        # self.progress = ProgressInterface(self)
        # """See `ProgressInterface()` """
        # self.MIP = MIPSolutionInterface(self)
        # """See `MIPSolutionInterface()` """
        # self.pool = SolnPoolInterface(self)
        # """See `SolnPoolInterface()` """

    def get_status(self):
        """Returns the status of the solution.

        Returns an attribute of Cplex.solution.status.
        For interpretations of the status codes, see the
        reference manual of the CPLEX Callable Library,
        especially the group optim.cplex.callable.solutionstatus

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.solve()
        >>> c.solution.get_status()
        1
        """
        raise NotImplementedError

    def get_method(self):
        """Returns the method used to solve the problem.

        Returns an attribute of Cplex.solution.method.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.solve()
        >>> c.solution.get_method()
        2
        """
        raise NotImplementedError

    def get_status_string(self, status_code=None):
        """Returns a string describing the status of the solution.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.solve()
        >>> c.solution.get_status_string()
        'optimal'
        """
        # pylint: disable=unused-argument
        # if status_code is None:
        #    status_code = self.get_status()
        raise NotImplementedError

    def get_objective_value(self):
        """Returns the value of the objective function.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.solve()
        >>> c.solution.get_objective_value()
        -202.5
        """
        raise NotImplementedError

    def get_values(self, *args):
        """Returns the values of a set of variables at the solution.

        Can be called by four forms.

        solution.get_values()
          return the values of all variables from the problem.

        solution.get_values(i)
          i must be a variable name or index.  Returns the value of
          the variable whose index or name is i.

        solution.get_values(s)
          s must be a sequence of variable names or indices.  Returns
          the values of the variables with indices the members of s.
          Equivalent to [solution.get_values(i) for i in s]

        solution.get_values(begin, end)
          begin and end must be variable indices or variable names.
          Returns the values of the variables with indices between begin
          and end, inclusive of end. Equivalent to
          solution.get_values(range(begin, end + 1)).

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> c.solution.get_values([0, 4, 5])
        [25.5, 0.0, 80.0]
        """
        # pylint: disable=unused-argument
        raise NotImplementedError

    def get_integer_quality(self, which):
        """Returns a measure of the quality of the solution.

        The measure of the quality of a solution can be a single attribute of
        solution.quality_metrics or a sequence of such
        attributes.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> m = c.solution.quality_metric
        >>> c.solution.get_integer_quality([m.max_x, m.max_dual_infeasibility])
        [18, -1]
        """
        # if isinstance(which, int):
        #     return None
        # else:
        #     return [None for a in which]
        raise NotImplementedError

    def get_float_quality(self, which):
        """Returns a measure of the quality of the solution.

        The measure of the quality of a solution can be a single attribute of
        solution.quality_metrics or a sequence of such attributes.

        Note
          This corresponds to the CPLEX callable library function
          CPXgetdblquality.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> m = c.solution.quality_metric
        >>> c.solution.get_float_quality([m.max_x, m.max_dual_infeasibility])
        [500.0, 0.0]
        """
        # if isinstance(which, int):
        #     return None
        # else:
        #     return [None for a in which]
        raise NotImplementedError

    def get_solution_type(self):
        """Returns the type of the solution.

        Returns an attribute of Cplex.solution.type.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> c.solution.get_solution_type()
        1
        """
        raise NotImplementedError

    def is_primal_feasible(self):
        """Returns whether or not the solution is known to be primal feasible.

        Note
          Returning False does not necessarily mean that the problem is
          not primal feasible, only that it is not proved to be primal
          feasible.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> c.solution.is_primal_feasible()
        True
        """
        raise NotImplementedError

    def get_quality_metrics(self):
        """Returns an object containing measures of the solution quality.

        See `QualityMetrics`
        """
        return QualityMetrics()

    def write(self, filename):
        """Writes the incumbent solution to a file.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> c.solution.write("lpex.sol")
        """
        # pylint: disable=unused-argument
        raise NotImplementedError

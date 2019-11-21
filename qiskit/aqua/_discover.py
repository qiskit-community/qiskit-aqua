# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Methods for pluggable objects discovery, registration, information
"""

import logging
import os
import pkgutil
import importlib
import inspect
import copy
from collections import namedtuple
from enum import Enum
import pkg_resources
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)

PLUGGABLES_ENTRY_POINT = 'qiskit.aqua.pluggables'


class PluggableType(Enum):
    """ Pluggable Types """
    ALGORITHM = 'algorithm'
    OPTIMIZER = 'optimizer'
    VARIATIONAL_FORM = 'variational_form'
    INITIAL_STATE = 'initial_state'
    IQFT = 'iqft'
    QFT = 'qft'
    ORACLE = 'oracle'
    FEATURE_MAP = 'feature_map'
    MULTICLASS_EXTENSION = 'multiclass_extension'
    UNCERTAINTY_PROBLEM = 'uncertainty_problem'
    UNIVARIATE_DISTRIBUTION = 'univariate_distribution'
    MULTIVARIATE_DISTRIBUTION = 'multivariate_distribution'
    INPUT = 'input'
    EIGENVALUES = 'eigs'
    RECIPROCAL = 'reciprocal'
    GENERATIVE_NETWORK = 'generative_network'
    DISCRIMINATIVE_NETWORK = 'discriminative_network'


def _get_pluggables_types_dict():
    """
    Gets all the pluggables types
    Any new pluggable type should be added here
    """
    # pylint: disable=import-outside-toplevel
    from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
    from qiskit.aqua.components.uncertainty_models import UnivariateDistribution
    from qiskit.aqua.components.uncertainty_models import MultivariateDistribution
    from qiskit.aqua.components.optimizers import Optimizer
    from qiskit.aqua.algorithms.quantum_algorithm import QuantumAlgorithm
    from qiskit.aqua.components.variational_forms import VariationalForm
    from qiskit.aqua.components.initial_states import InitialState
    from qiskit.aqua.components.iqfts import IQFT
    from qiskit.aqua.components.qfts import QFT
    from qiskit.aqua.components.oracles import Oracle
    from qiskit.aqua.components.feature_maps import FeatureMap
    from qiskit.aqua.components.multiclass_extensions import MulticlassExtension
    from qiskit.aqua.input import AlgorithmInput
    from qiskit.aqua.components.eigs import Eigenvalues
    from qiskit.aqua.components.reciprocals import Reciprocal
    from qiskit.aqua.components.neural_networks.discriminative_network import \
        DiscriminativeNetwork
    from qiskit.aqua.components.neural_networks.generative_network import GenerativeNetwork

    return {
        PluggableType.ALGORITHM: QuantumAlgorithm,
        PluggableType.OPTIMIZER: Optimizer,
        PluggableType.VARIATIONAL_FORM: VariationalForm,
        PluggableType.INITIAL_STATE: InitialState,
        PluggableType.IQFT: IQFT,
        PluggableType.QFT: QFT,
        PluggableType.ORACLE: Oracle,
        PluggableType.FEATURE_MAP: FeatureMap,
        PluggableType.MULTICLASS_EXTENSION: MulticlassExtension,
        PluggableType.UNCERTAINTY_PROBLEM: UncertaintyProblem,
        PluggableType.UNIVARIATE_DISTRIBUTION: UnivariateDistribution,
        PluggableType.MULTIVARIATE_DISTRIBUTION: MultivariateDistribution,
        PluggableType.INPUT: AlgorithmInput,
        PluggableType.EIGENVALUES: Eigenvalues,
        PluggableType.RECIPROCAL: Reciprocal,
        PluggableType.DISCRIMINATIVE_NETWORK: DiscriminativeNetwork,
        PluggableType.GENERATIVE_NETWORK: GenerativeNetwork
    }


_NAMES_TO_EXCLUDE = [os.path.basename(__file__)]

_FOLDERS_TO_EXCLUDE = ['__pycache__', 'gauopen']


class DiscoverRegistry:
    """Contains Discovered Classes Info."""

    REGISTERED_INFO = namedtuple(
        'REGISTERED_INFO', ['name', 'cls', 'configuration'])

    def __init__(self) -> None:
        self._discovered = False
        self._registry = {}

    @property
    def discovered(self):
        """ Returns discovered flag """
        return self._discovered

    def set_discovered(self):
        """ Set registry as discovered """
        self._discovered = True

    @property
    def registry(self):
        """ Return registry dictionary """
        return self._registry

    def reset(self):
        """ reset registry data """
        self._discovered = False
        self._registry = {}


# Registry Global Instance
_REGISTRY_PLUGGABLE = DiscoverRegistry()


def refresh_pluggables():
    """
    Attempts to rediscover all pluggable modules
    """
    _REGISTRY_PLUGGABLE.reset()
    _REGISTRY_PLUGGABLE.set_discovered()
    directory = os.path.dirname(__file__)
    _discover_local_pluggables(directory)
    _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'ml')),
                               'qiskit.ml')
    _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'chemistry')),
                               'qiskit.chemistry')
    _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'finance')),
                               'qiskit.finance')
    _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'optimization')),
                               'qiskit.optimization')
    _discover_entry_pt_pluggables()
    if logger.isEnabledFor(logging.DEBUG):
        for ptype in local_pluggables_types():
            logger.debug("Found: '%s' has pluggables %s ", ptype.value, local_pluggables(ptype))


def _discover_on_demand():
    """
    Attempts to discover pluggable modules, if not already discovered
    """
    if not _REGISTRY_PLUGGABLE.discovered:
        _REGISTRY_PLUGGABLE.reset()
        _REGISTRY_PLUGGABLE.set_discovered()
        directory = os.path.dirname(__file__)
        _discover_local_pluggables(directory)
        _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'ml')),
                                   'qiskit.ml')
        _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'chemistry')),
                                   'qiskit.chemistry')
        _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'finance')),
                                   'qiskit.finance')
        _discover_local_pluggables(os.path.abspath(os.path.join(directory, '..', 'optimization')),
                                   'qiskit.optimization')
        _discover_entry_pt_pluggables()
        if logger.isEnabledFor(logging.DEBUG):
            for ptype in local_pluggables_types():
                logger.debug("Found: '%s' has pluggables %s ", ptype.value, local_pluggables(ptype))


def _discover_entry_pt_pluggables():
    """
    Discovers the pluggable modules defined by entry_points in setup
    and attempts to register them. Pluggable modules should subclass Pluggable Base classes.
    """
    for entry_point in pkg_resources.iter_entry_points(PLUGGABLES_ENTRY_POINT):
        # first calls require and log any errors returned due to dependencies mismatches
        try:
            entry_point.require()
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning("Entry point '%s' requirements issue: %s", entry_point, str(ex))

        # now  call resolve and try to load entry point
        try:
            e_p = entry_point.resolve()
            _registered = False
            for pluggable_type, c in _get_pluggables_types_dict().items():
                if not inspect.isabstract(e_p) and issubclass(e_p, c):
                    _register_pluggable(pluggable_type, e_p)
                    _registered = True
                    logger.debug(
                        "Registered entry point pluggable type '%s' '%s' class '%s'",
                        pluggable_type.value, entry_point, e_p)
                    break

            if not _registered:
                # print("Unknown entry point pluggable '{}' class '{}'".format(entry_point, e_p))
                logger.debug("Unknown entry point pluggable '%s' class '%s'", entry_point, e_p)
        except Exception as ex:  # pylint: disable=broad-except
            # Ignore entry point that could not be initialized.
            # print("Failed to load entry point '{}' error {}".format(entry_point, str(e)))
            logger.debug("Failed to load entry point '%s' error %s", entry_point, str(ex))


def _discover_local_pluggables(directory=os.path.dirname(__file__),
                               parentname=os.path.splitext(__name__)[0],
                               names_to_exclude=None,
                               folders_to_exclude=None):
    """
    Discovers the pluggable modules on the directory and subdirectories of the current module
    and attempts to register them. Pluggable modules should subclass Pluggable Base classes.
    Args:
        directory (Optional(str)): Directory to search for pluggable. Defaults
            to the directory of this module.
        parentname (Optional(str)): Module parent name. Defaults to current directory name
        names_to_exclude (Optional(list[str])): File names to exclude.
                                                    Defaults to _NAMES_TO_EXCLUDE
        folders_to_exclude (Optional(list[str])): Folders to exclude.
                                                    Defaults to _FOLDERS_TO_EXCLUDE
    """
    names_to_exclude = names_to_exclude if names_to_exclude is not None else _NAMES_TO_EXCLUDE
    folders_to_exclude = folders_to_exclude \
        if folders_to_exclude is not None else _FOLDERS_TO_EXCLUDE
    for _, name, ispackage in pkgutil.iter_modules([directory]):
        if ispackage:
            continue

        # Iterate through the modules
        if name not in names_to_exclude:  # skip those modules
            try:
                fullname = parentname + '.' + name
                modspec = importlib.util.find_spec(fullname)
                mod = importlib.util.module_from_spec(modspec)
                modspec.loader.exec_module(mod)
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    # Iterate through the classes defined on the module.
                    try:
                        if cls.__module__ == modspec.name:
                            for pluggable_type, c in _get_pluggables_types_dict().items():
                                if not inspect.isabstract(cls) and issubclass(cls, c):
                                    _register_pluggable(pluggable_type, cls)
                                    importlib.import_module(fullname)
                                    break
                    except Exception as ex:  # pylint: disable=broad-except
                        # Ignore pluggables that could not be initialized.
                        # print('Failed to load pluggable {} error {}'.format(fullname, str(ex)))
                        logger.debug('Failed to load pluggable %s error %s', fullname, str(ex))

            except Exception as ex:  # pylint: disable=broad-except
                # Ignore pluggables that could not be initialized.
                # print('Failed to load {} error {}'.format(fullname, str(ex)))
                logger.debug('Failed to load %s error %s', fullname, str(ex))

    for item in sorted(os.listdir(directory)):
        fullpath = os.path.join(directory, item)
        if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
            _discover_local_pluggables(fullpath, parentname + '.' + item,
                                       names_to_exclude, folders_to_exclude)


def register_pluggable(cls):
    """
    Registers a pluggable class
    Args:
        cls (Pluggable): Pluggable class.
     Returns:
        str: pluggable name
     Raises:
         AquaError: Class doesn't derive from known pluggable
    """
    _discover_on_demand()
    pluggable_type = None
    for p_type, c in _get_pluggables_types_dict().items():
        if issubclass(cls, c):
            pluggable_type = p_type
            break

    if pluggable_type is None:
        raise AquaError(
            'Could not register class {} is not subclass of any known pluggable'.format(cls))

    return _register_pluggable(pluggable_type, cls)


GLOBAL_CLASS = None


def _register_pluggable(pluggable_type, cls):
    """
    Registers a pluggable class
    Args:
        pluggable_type(PluggableType): The pluggable type
        cls (Pluggable): Pluggable class.
     Returns:
        str: pluggable name
    Raises:
        AquaError: if the class is already registered or could not be registered
    """
    if pluggable_type not in _REGISTRY_PLUGGABLE.registry:
        _REGISTRY_PLUGGABLE.registry[pluggable_type] = {}

    # fix pickle problems
    method = 'from {} import {}\nglobal GLOBAL_CLASS\nGLOBAL_CLASS = {}'.format(
                cls.__module__, cls.__qualname__, cls.__qualname__)
    exec(method)  # pylint: disable=exec-used
    cls = GLOBAL_CLASS

    # Verify that the pluggable is not already registered.
    registered_classes = _REGISTRY_PLUGGABLE.registry[pluggable_type]
    if cls in [pluggable.cls for pluggable in registered_classes.values()]:
        raise AquaError(
            'Could not register class {} is already registered'.format(cls))

    # Verify that it has a minimal valid configuration.
    try:
        pluggable_name = cls.CONFIGURATION['name']
    except (LookupError, TypeError):
        raise AquaError('Could not register pluggable: invalid configuration')

    # Verify that the pluggable is valid
    check_pluggable_valid = getattr(cls, 'check_pluggable_valid', None)
    if check_pluggable_valid is not None:
        try:
            # pylint: disable=not-callable
            check_pluggable_valid()
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug(str(ex))
            raise AquaError('Could not register class {}. Name {} is not valid'.format(
                cls, pluggable_name)) from ex

    if pluggable_name in _REGISTRY_PLUGGABLE.registry[pluggable_type]:
        raise AquaError('Could not register class {}. Name {} {} '
                        'is already registered'.format(
                            cls, pluggable_name,
                            _REGISTRY_PLUGGABLE.registry[pluggable_type][pluggable_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTRY_PLUGGABLE.registry[pluggable_type][pluggable_name] = DiscoverRegistry.REGISTERED_INFO(
        pluggable_name, cls, copy.deepcopy(cls.CONFIGURATION))
    return pluggable_name


def deregister_pluggable(pluggable_type, pluggable_name):
    """
    Deregisters a pluggable class
    Args:
        pluggable_type(PluggableType): The pluggable type
        pluggable_name (str): The pluggable name
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTRY_PLUGGABLE.registry:
        raise AquaError('Could not deregister {} {} not registered'.format(
            pluggable_type, pluggable_name))

    if pluggable_name not in _REGISTRY_PLUGGABLE.registry[pluggable_type]:
        raise AquaError('Could not deregister {} {} not registered'.format(
            pluggable_type, pluggable_name))

    _REGISTRY_PLUGGABLE.registry[pluggable_type].pop(pluggable_name)


def get_pluggable_class(pluggable_type, pluggable_name):
    """
    Accesses pluggable class
    Args:
        pluggable_type(Union(PluggableType,str)): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        Pluggable: pluggable class
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError('Invalid pluggable type {} {}'.format(pluggable_type, pluggable_name))

    if pluggable_type not in _REGISTRY_PLUGGABLE.registry:
        raise AquaError('{} {} not registered'.format(pluggable_type, pluggable_name))

    if not pluggable_name:
        raise AquaError(
            'Unable to get class for pluggable {}: Missing name.'.format(pluggable_type))

    if pluggable_name not in _REGISTRY_PLUGGABLE.registry[pluggable_type]:
        raise AquaError("{} '{}' not registered".format(pluggable_type, pluggable_name))

    return _REGISTRY_PLUGGABLE.registry[pluggable_type][pluggable_name].cls


def get_pluggable_configuration(pluggable_type, pluggable_name):
    """
    Accesses pluggable configuration
    Args:
        pluggable_type(Union(PluggableType,str)): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        dict: pluggable configuration
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError('Invalid pluggable type {} {}'.format(pluggable_type, pluggable_name))

    if pluggable_type not in _REGISTRY_PLUGGABLE.registry:
        raise AquaError('{} {} not registered'.format(pluggable_type, pluggable_name))

    if not pluggable_name:
        raise AquaError(
            'Unable to get configuration for pluggable {}: Missing name.'.format(pluggable_type))

    if pluggable_name not in _REGISTRY_PLUGGABLE.registry[pluggable_type]:
        raise AquaError('{} {} not registered'.format(pluggable_type, pluggable_name))

    return copy.deepcopy(_REGISTRY_PLUGGABLE.registry[pluggable_type][pluggable_name].configuration)


def local_pluggables_types():
    """
    Accesses all pluggable types
    Returns:
       list[PluggableType]: pluggable types
    """
    _discover_on_demand()
    return list(_REGISTRY_PLUGGABLE.registry.keys())


def local_pluggables(pluggable_type):
    """
    Accesses pluggable names
    Args:
        pluggable_type (Union(PluggableType,str)): The pluggable type
    Returns:
        list[str]: pluggable names
    Raises:
        AquaError: if the type is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError(
            'Invalid pluggable type {}'.format(pluggable_type))

    if pluggable_type not in _REGISTRY_PLUGGABLE.registry:
        raise AquaError('{} not registered'.format(pluggable_type))

    return [pluggable.name for pluggable in _REGISTRY_PLUGGABLE.registry[pluggable_type].values()]

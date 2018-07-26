Contributing to Qiskit Aqua
============================

Qiskit Aqua has a modular and extensible architecture.

Instead of just *accessing* Qiskit Aqua as a library of quantum algorithms to experiment with quantum
computing, a user may decide to *contribute* to Qiskit Aqua by
providing new algorithms and algorithm components.
These can be programmatically added to Qiskit Aqua,
which was designed as an extensible, pluggable
framework.

.. topic:: Contribution Guidelines

    Any user who would like to contribute to Qiskit Aqua should follow the Qiskit Aqua `contribution
    guidelines <https://github.com/Qiskit/aqua/blob/master/.github/CONTRIBUTING.rst>`__.

Extending Qiskit Aqua
----------------------

Qiskit Aqua exposes numerous extension points. Researchers and developers can contribute to Qiskit Aqua
by providing new components, which will be automatically discovered and loaded by Qiskit Aqua at run time.

Dynamically Discovered Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component should derive from the corresponding base class, as explained below.  There are three
ways for a component to be dynamically discovered and loaded by Qiskit Aqua at run time:

1. The class implementing the component should be placed in the appropriate folder in the file system,
   as explained in `Section "Extension Points" <#extension-points>`__ below for each different component type.
   This is the easiest approach.  Researchers
   and developers extending Qiskit Aqua are more likely to have installed Qiskit Aqua by cloning the
   `Qiskit Aqua repository <https://github.com/Qiskit/aqua>`__ as opposed to using the pip package
   manager system.  Therefore, the folders indicated below can be easily located in the file system.

2. Alternatively, a developer extending Qiskit Aqua with a new component can simply create a dedicated
   repository with its own versioning.  This repository must be locally installable with the package that was
   created.  Once the repository has been installed, for example via the ``pip install -e`` command,
   the user can access the
   Qiskit Aqua `Graphical User Interface (GUI) <https://qiskit.org/documentation/aqua/install.html#gui>`__
   and add the package's name to the list of packages in the **Preferences** panel.
   From that moment on, any custom component found below that package will be dynamically added to
   ``qiskit-aqua`` upon initialization.

3. There is yet another way to achieve the same goal, and it simply consists of customizing the
   ``setup.py`` file of the new component in order to add the package's name to ``qiskit-aqua``
   when someone installs the package, without the need of using the GUI to enter it later.  This is an example
   of what ``setup.py`` would look like:

   .. code:: python

       import setuptools
       from setuptools.command.install import install
       from setuptools.command.develop import develop
       from setuptools.command.egg_info import egg_info
       import atexit

       long_description = """New Package for Qiskit Aqua Component"""
    
       requirements = [
          "qiskit-aqua>=0.2.0",
          "qiskit>=0.5.6",
          "numpy>=1.13,<1.15"
       ]

       def _post_install():
          from qiskit_aqua.preferences import Preferences
          preferences = Preferences()
          preferences.add_package('aqua_custom_component_package')
          preferences.save()

       class CustomInstallCommand(install):
          def run(self):
          atexit.register(_post_install)
          install.run(self)
        
       class CustomDevelopCommand(develop):
          def run(self):
          atexit.register(_post_install)
          develop.run(self)
        
       class CustomEggInfoCommand(egg_info):
          def run(self):
          atexit.register(_post_install)
          egg_info.run(self)
    
       setuptools.setup(
          name = 'aqua_custom_component_package',
          version = "0.1.0", # this should match __init__.__version__
          description='Qiskit Aqua Component',
          long_description = long_description,
          long_description_content_type = "text/markdown",
          url = 'https://github.com/aqua-custom-component-package',
          author = 'Qiskit Aqua Development Team',
          author_email = 'qiskit@us.ibm.com',
          license='Apache-2.0',
          classifiers = (
             "Environment :: Console",
             "License :: OSI Approved :: Apache Software License",
             "Intended Audience :: Developers",
             "Intended Audience :: Science/Research",
             "Operating System :: Microsoft :: Windows",
             "Operating System :: MacOS",
             "Operating System :: POSIX :: Linux",
             "Programming Language :: Python :: 3.5",
             "Programming Language :: Python :: 3.6",
             "Topic :: Scientific/Engineering"
          ),
          keywords = 'qiskit sdk quantum aqua',
          packages = setuptools.find_packages(exclude=['test*']),
          install_requires = requirements,
          include_package_data = True,
          python_requires = ">=3.5",
          cmdclass = {
             'install': CustomInstallCommand,
             'develop': CustomDevelopCommand,
             'egg_info': CustomEggInfoCommand
          }
       )

.. note::
    All the classes implementing the algorithms and the supporting components listed below
    should embed a configuration dictionary including ``name``, ``description`` and ``input_schema`` properties.

Extension Points
~~~~~~~~~~~~~~~~

This section details the algorithm and algorithm components that researchers and developers
interested in quantum algorithms can contribute to Qiskit Aqua.

Algorithms
^^^^^^^^^^

A new `algorithm <./algorithms.html>`__ may be developed according to the specific API provided by Qiskit Aqua.
By simply adding its code to the collection of existing algorithms, that new algorithm
will be immediately recognized via dynamic lookup, and made available for use within the framework of Qiskit Aqua.
To develop and deploy any new algorithm, the new algorithm class should derive from the ``QuantumAlgorithm`` class.
Along with all of its supporting modules, the new algorithm class should be installed under its own folder in the
``qiskit_aqua`` directory, just like the existing algorithms.

Optimizers
^^^^^^^^^^

New `optimizers <./optimizers.html>`__ for quantum variational algorithms
should be installed in the ``qiskit_aqua/utils/optimizers`` folder  and derive from
the ``Optimizer`` class.

Variational Forms
^^^^^^^^^^^^^^^^^

`Trial wavefunctions <./variational_forms.html>`__ for quantum variational algorithms, such as
`VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__
should go under the ``qiskit_aqua/utils/variational_forms`` folder
and derive from the ``VariationalForm`` class.

Initial States
^^^^^^^^^^^^^^

`Initial states <./initial_states.html>`__, for algorithms such as `VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__,
`QPE <./algorithms.html#quantum-phase-estimation-qpe>`__
and `IQPE <./algorithms.html#iterative-quantum-phase-estimation-iqpe>`__, should go under the ``qiskit_aqua/utils/initial_states`` folder and
derive from the ``InitialState`` class.

Inverse Quantum Fourier Transforms (IQFTs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`IQFTs <./iqfts.html>`__, for use for example for `QPE <./algorithms.html#quantum-phase-estimation-qpe>`__, should be installed  under the
``qiskit_aqua/utils/iqfts`` folder and derive from the ``IQFT`` class.

Oracles
^^^^^^^

`Oracles <./oracles.html>`__, for use with algorithms such as `Grover's search <./algorithms.html#quantum-grover-search>`__,
should go under the
``qiskit_aqua/utils/oracles`` folder  and derive from the ``Oracle`` class.

Unit Tests
----------

Contributing new software components to Qiskit Aqua requires writing new unit tests for those components,
and executing all the existing unit tests to make sure that no bugs were inadvertently injected.


Writing Unit Tests
~~~~~~~~~~~~~~~~~~
Unit tests should go under the ``test`` folder and be classes derived from
the ``QISKitAquaTestCase`` class.  They should not have ``print`` statements;
rather, they should use ``self.log.debug``. If
they use assertions, these should be from the ``unittest`` package, such as
``self.AssertTrue``, ``self.assertRaises``, etc.

Executing Unit Tests
~~~~~~~~~~~~~~~~~~~~
To run all unit tests, execute the following command:

.. code:: sh

    python -m unittest discover

To run a particular unit test module, the following command should be used:

.. code:: sh

    python -m unittest test/test_end2end.py

The command for help is as follows:

.. code::

    python -m unittest -h

`Other running options <https://docs.python.org/3/library/unittest.html#command-line-options>`__ are available
to users for consultation.

In order to see unit test log messages, researchers and developers contributing to Qiskit Aqua
will need to set the ``LOG_LEVEL`` environment variable to ``DEBUG`` mode:

.. code:: sh

    LOG_LEVEL=DEBUG
    export LOG_LEVEL

The results from ``self.log.debug`` will be saved to a
file with same name as the module used to run, and with a ``log`` extension. For instance,
the ``test_end2end.py`` script in the example above will generate a log file named
``test_end2end.log`` in the ``test`` folder.

.. _aqua-extending:

====================
Contributing to Aqua
====================

Aqua has a modular and extensible architecture.
Instead of just *accessing* Aqua as a library of quantum algorithms to experiment with quantum
computing, a user may decide to *contribute* to Aqua by
providing new algorithms and algorithm components.
These can be programmatically added to Aqua,
which was designed as an extensible, pluggable
framework.

.. topic:: Contribution Guidelines

    Any user who would like to contribute to Aqua should follow the Aqua `contribution
    guidelines <https://github.com/QISKit/aqua/blob/master/.github/CONTRIBUTING.rst>`__.

--------------
Extending Aqua
--------------

Aqua exposes numerous extension points. Researchers and developers can contribute to Aqua
by providing new components, which will be automatically discovered and loaded by Aqua at run time.

.. _aqua-dynamically-discovered-components:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamically Discovered Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each component should derive from the corresponding base class, as explained below.  There are three
ways for a component to be dynamically discovered and loaded by Aqua at run time:

1. The class implementing the component should be placed in the appropriate folder in the file system,
   as explained in `Section "Aqua Extension Points" <#aqua-extension-points>`__ below for each different component type.
   This is the easiest approach.  Researchers
   and developers extending Aqua are more likely to have installed Aqua by cloning the
   `Aqua repository <https://github.com/Qiskit/aqua>`__ as opposed to using the pip package
   manager system.  Therefore, the folders indicated below can be easily located in the file system.

2. Alternatively, a developer extending Aqua with a new component can simply create a dedicated
   repository with its own versioning.  This repository must be locally installable with the package that was
   created.  Once the repository has been installed, for example via the ``pip install -e`` command,
   the user can access the
   Aqua `Graphical User Interface (GUI) <#aqua-gui>`__
   and add the package's name to the list of packages in the **Preferences** panel.
   From that moment on, any custom component found below that package will be dynamically added to
   ``aqua`` upon initialization.

3. There is yet another way to achieve the same goal, and it simply consists of customizing the
   ``setup.py`` file of the new component in order to add the package's name to ``aqua``
   when someone installs the package, without the need of using the GUI to enter it later.  This is an example
   of what ``setup.py`` would look like:

   .. code:: python

       import setuptools
       from setuptools.command.install import install
       from setuptools.command.develop import develop
       from setuptools.command.egg_info import egg_info
       import atexit

       long_description = """New Package for Aqua Component"""
    
       requirements = [
          "aqua>=0.1.2",
          "qiskit>=0.5.6",
          "numpy>=1.13,<1.15"
       ]

       def _post_install():
          from qiskit_acqua.preferences import Preferences
          preferences = Preferences()
          preferences.add_package('acqua_custom_component_package')
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
          name = 'acqua_custom_component_package',
          version = "0.1.0", # this should match __init__.__version__
          description='Aqua Component',
          long_description = long_description,
          long_description_content_type = "text/markdown",
          url = 'https://github.com/acqua-custom-component-package',
          author = 'Aqua Development Team',
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
          keywords = 'qiskit sdk quantum acqua',
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

^^^^^^^^^^^^^^^^^^^^^
Aqua Extension Points
^^^^^^^^^^^^^^^^^^^^^

This section details the algorithm and algorithm components that researchers and developers
interested in quantum algorithms can contribute to Aqua.

.. _extending-algorithms:

^^^^^^^^^^
Algorithms
^^^^^^^^^^

New :ref:`quantum-algorithms` may be developed according to the specific API provided by Aqua.
By simply adding the code of an algorithm to the collection of existing algorithms, that new algorithm
will be immediately recognized via dynamic lookup, and made available for use within the framework of Aqua.
To develop and deploy any new algorithm, the new algorithm class should derive from the ``QuantumAlgorithm`` class.
Along with all of its supporting modules, the new algorithm class should be installed under its own folder in the
``qiskit_aqua`` directory, just like the existing algorithms, unless the dynamic-discovery approach has been
chosen, in which case the algorithm can register itself as an Aqua algorithm irrespective of its installation
folder in the file system.

.. _extending-optimizers:

^^^^^^^^^^
Optimizers
^^^^^^^^^^

New `optimizers <#optimizers>`__ for quantum variational algorithms
should and derive from
the ``Optimizer`` class.  They should also be installed in the ``qiskit_aqua/utils/optimizers`` folder
of the ``aqua`` repository clone,
unless the dynamic-discovery approach has been
chosen, in which case a new optimizer can register itself as an Aqua optimizer irrespective of its installation
folder in the file system.

.. _extending-variational-forms:

^^^^^^^^^^^^^^^^^
Variational Forms
^^^^^^^^^^^^^^^^^

`Trial wave functions <#variational_forms>`__ for quantum variational algorithms, such as
`VQE <#variational-quantum-eigensolver-vqe>`__ must derive from the ``VariationalForm`` class.
They should also be installed under the ``qiskit_aqua/utils/variational_forms`` folder
unless the dynamic-discovery approach has been
chosen, in which case a new trial wave function can register itself as an Aqua variational form irrespective of its installation
folder in the file system.

.. _extending-oracles:

^^^^^^^
Oracles
^^^^^^^

`Oracles <#oracles>`__, for use with algorithms such as `Grover's search <#quantum-grover-search>`__,
should derive from the ``Oracle`` class.  They should also go under the
``qiskit_aqua/utils/oracles`` folder, 
unless the dynamic-discovery approach has been
chosen, in which case a new oracle can register itself as an Aqua oracle irrespective of its installation
folder in the file system.

.. _extending-iqfts:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Inverse Quantum Fourier Transforms (IQFTs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`IQFTs <#iqfts>`__, for use for example for `QPE <#quantum-phase-estimation-qpe>`__, must derive from the ``IQFT`` class.
They should also be installed  under the ``qiskit_aqua/utils/iqfts`` folder,
unless the dynamic-discovery approach has been
chosen, in which case a new IQFT can register itself as an Aqua IQFT irrespective of its installation
folder in the file system.

.. _extending-initial-states:

^^^^^^^^^^^^^^
Initial States
^^^^^^^^^^^^^^

`Initial states <#initial_states>`__, for algorithms such as `VQE <#variational-quantum-eigensolver-vqe>`__,
`QPE <#quantum-phase-estimation-qpe>`__
and `IQPE <#iterative-quantum-phase-estimation-iqpe>`__, must derive from the ``InitialState`` class.
They should also be installed under the ``qiskit_aqua/utils/initial_states`` folder,
unless the dynamic-discovery approach has been
chosen, in which case a new initial state can register itself as an Aqua initial state irrespective of its installation
folder in the file system.

---------------------
Aqua Documentation UI
---------------------
Researchers and developers interested in extending Aqua with new
algorithms and computational components can access the :ref:`aqua-doc-ui`,
which offers a quick and succinct overview of all the extensible components
along with their configuration schemas.

---------------
Aqua Unit Tests
---------------

Contributing new software components to Aqua requires writing new unit tests for those components,
and executing all the existing unit tests to make sure that no bugs were inadvertently injected.

^^^^^^^^^^^^^^^^^^^^^^^
Writing Aqua Unit Tests
^^^^^^^^^^^^^^^^^^^^^^^

Unit tests should go under the ``test`` folder and be classes derived from
the ``QiskitAquaTestCase`` class.  They should not have ``print`` statements;
rather, they should use ``self.log.debug``. If
they use assertions, these should be from the ``unittest`` package, such as
``self.AssertTrue``, ``self.assertRaises``, etc.

^^^^^^^^^^^^^^^^^^^^^^^^^
Executing Aqua Unit Tests
^^^^^^^^^^^^^^^^^^^^^^^^^
To run all unit tests, execute the following command:

.. code:: sh

    python -m unittest discover

To run a particular unit test module, the following command should be used:

.. code:: sh

    python -m unittest test/test_end2end.py

The command for help is as follows:

.. code::

    python -m unittest -h

.. seealso::
    `Other running options <https://docs.python.org/3/library/unittest.html#command-line-options>`__ are available
    to users for consultation.

In order to see unit test log messages, researchers and developers contributing to Aqua
will need to set the ``LOG_LEVEL`` environment variable to ``DEBUG`` mode:

.. code:: sh

    LOG_LEVEL=DEBUG
    export LOG_LEVEL

The results from ``self.log.debug`` will be saved to a
file with same name as the module used to run, and with a ``log`` extension. For instance,
the ``test_end2end.py`` script in the example above will generate a log file named
``test_end2end.log`` in the ``test`` folder.

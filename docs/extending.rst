Contributing to QISKit ACQUA Chemistry
======================================

QISKit ACQUA Chemistry, just like the QISKit ACQUA library it is built upon, has a modular and extensible architecture.

Instead of just *accessing* QISKit ACQUA Chemistry as a library of quantum algorithms and tools to experiment with quantum
computing for chemistry, a user may decide to *contribute* to QISKit ACQUA Chemistry by
providing new components.
These can be programmatically added to QISKit ACQUA Chemistry,
which was designed as an extensible, pluggable
framework.  Once added, new components are automatically discovered.

.. topic:: Contribution Guidelines

    Any user who would like to contribute to QISKit ACQUA should follow the QISKit ACQUA `contribution
    guidelines <https://github.com/QISKit/qiskit-acqua-chemistry/blob/master/.github/CONTRIBUTING.rst>`__.

Extending QISKit ACQUA Chemistry
--------------------------------

QISKit ACQUA Chemistry exposes two extension points. Researchers and developers can contribute to QISKit ACQUA Chemistry
by providing new components, which will be automatically discovered and loaded by QISKit ACQUA at run time.

Dynamically Discovered Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component should derive from the corresponding base class, as explained below.  There are three
ways for a component to be dynamically discovered and loaded by QISKit ACQUA Chemistry at run time:

1. The class implementing the component should be placed in the appropriate folder in the file system,
   as explained in `Section "Extension Points" <#extension-points>`__ below for each different component type.
   This is the easiest approach.  Researchers
   and developers extending QISKit ACQUA Chemistry are more likely to have installed QISKit ACQUA Chemistry by cloning the
   `QISKit ACQUA Chemistry repository <https://github.com/Qiskit/qiskit-acqua-chemistry>`__ as opposed to using
   the pip package manager system.  Therefore, the folders indicated below can be easily located in the file system.

2. Alternatively, a developer extending QISKit ACQUA Chemistry with a new component can simply create a dedicated
   repository with its own versioning.  This repository must be locally installable with the package that was
   created.  Once the repository has been installed, for example via the ``pip install -e`` command,
   the user can access the
   QISKit ACQUA Chemistry `Graphical User Interface (GUI) <https://qiskit.org/documentation/acqua/chemistry/install.html#gui>`__
   and add the package's name to the list of packages in the **Preferences** panel.
   From that moment on, any custom component found below that package will be dynamically added to
   ``qiskit-acqua-chemistry`` upon initialization.

3. There is yet another way to achieve the same goal, and it simply consists of customizing the
   ``setup.py`` file of the new component in order to add the package's name to ``qiskit-acqua-chemistry``
   when someone installs the package, without the need of using the GUI to enter it later.  This is an example
   of what ``setup.py`` would look like:

   .. code:: python

       import setuptools
       from setuptools.command.install import install
       from setuptools.command.develop import develop
       from setuptools.command.egg_info import egg_info
       import atexit

       long_description = """New Package for QISKit ACQUA Chemistry Component"""
    
       requirements = [
          "qiskit-acqua-chemistry>=0.1.2",
          "qiskit>=0.5.6",
          "numpy>=1.13,<1.15"
       ]

       def _post_install():
          from qiskit_acqua_chemistry.preferences import Preferences
          preferences = Preferences()
          preferences.add_package('acqua_chemistry_custom_component_package')
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
          name = 'acqua_chemistry_custom_component_package',
          version = "0.1.0", # this should match __init__.__version__
          description='QISKit ACQUA Chemistry Component',
          long_description = long_description,
          long_description_content_type = "text/markdown",
          url = 'https://github.com/acqua-chemistry-custom-component-package',
          author = 'QISKit ACQUA Development Team',
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


Extension Points
~~~~~~~~~~~~~~~~
This section details the components that researchers and developers
can contribute to QISKit ACQUA Chemistry.

Drivers
^^^^^^^

The driver support in QISKit ACQUA Chemistry was designed to make the drivers pluggable and discoverable.
In order for QISKit ACQUA Chemistry to
be able to interface a driver library, the ``BaseDriver`` base class must be implemented in order
to provide the interfacing code, or *wrapper*.  As part of this process, the required
`JavaScript Object Notation (JSON) <http://json.org>`__ schema for the driver interface must
be provided in a file named ``configuration.json``.  The interfacing code in the driver wrapper
is responsible for constructing and populating a ``QMolecule`` instance with the electronic
structure data listed above.  Driver wrappers implementing the ``BaseDriver`` class and the
associated ``configuration.json`` schema file are organized in subfolders of the ``drivers`` folder
for automatic discovery and dynamic lookup.


Chemistry Operators
^^^^^^^^^^^^^^^^^^^

Chemistry operators convert the electronic structure information obtained from the
drivers to qubit-operator forms, suitable to be processed by
an `algorithm <https://qiskit.org/documentation/acqua/algorithms.html>`__ in QISKit ACQUA.  New chemistry operators
can be plugged in by extending the ``ChemistryOperator`` interface and providing the required
`JavaScript Object Notation (JSON) <>`__ schema.  Chemistry operator implementations are collected in the ``core`` folder
for automatic discovery and dynamic lookup.


Unit Tests
----------

Contributing new software components to QISKit ACQUA Chemistry requires writing new unit tests for those components,
and executing all the existing unit tests to make sure that no bugs were inadvertently injected.


Writing Unit Tests
~~~~~~~~~~~~~~~~~~
Unit tests should go under the ``test`` folder and be classes derived from
the ``QISKitAcquaChemistryTestCase`` class.  They should not have ``print`` statements;
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

In order to see unit test log messages, researchers and developers contributing to QISKit ACQUA
will need to set the ``LOG_LEVEL`` environment variable to ``DEBUG`` mode:

.. code:: sh

    LOG_LEVEL=DEBUG
    export LOG_LEVEL

The results from ``self.log.debug`` will be saved to a
file with same name as the module used to run, and with a ``log`` extension. For instance,
the ``test_end2end.py`` script in the example above will generate a log file named
``test_end2end.log`` in the ``test`` folder.
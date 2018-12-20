# Qiskit Chemistry

## Electronic structure drivers

Qiskit Chemistry requires a computational chemistry program or library to be available in
order that it can be used for electronic structure computation. For example the computation of one and two electron
integrals for the molecule in the experiment.

This folder contains drivers which have been already written to interface to a number of such programs/libraries. More
information for each driver about the program/library it interfaces with and installation instructions may be found in
each driver folder.

At least one chemistry program/library needs to be installed.

* [Gaussian](./gaussiand/README.md): A commercial chemistry program  
* [PyQuante](./pyquanted/README.md): An open-source Python library, with a pure Python version usable cross-platform
* [PySCF](./pyscfd/README.md): An open-source Python library 
* [PSI4](./psi4d/README.md): An open-source chemistry program built on Python

However it is possible to run some chemistry experiments if you have a Qiskit Chemistry HDF5 file that has been
previously created when using one of the above drivers. The HDF5 driver takes such an input. 

* [HDF5](./hdf5d/README.md): Driver for Qiskit Chemistry hdf5 files    

## Writing a new driver

The drivers here were designed to be pluggable and discoverable. Thus a new driver can be created and simply added and
will be found for use within Qiskit Chemistry. If you are writing a new driver to your favorite chemistry
program/library then the driver should derive from BaseDriver class.

A configuration.json file is also needed that names the driver and specifies its main class that has been
derived from BaseDriver.

The core of the driver should use the chemistry program/library and populate a QMolecule instance with the electronic
structure data.
 
Consulting the existing drivers may be helpful in accomplishing the above.

# IBM Quantum Library for Chemistry

## Electronic structure drivers

IBM Quantum Library for Chemistry, qiskit_acqua_chemistry, requires a computational chemistry program or library to be available in
order that it can be used for electronic structure computation. For example the computation of one and two electron
integrals for the molecule in the experiment.

This folder contains drivers which have been already written to interface to a number of such programs/libraries. More
information for each driver about the program/library it interfaces with and installation instructions may be found in
each driver folder.

At least one chemistry program/library needs to be installed. A number of different options are available here.

However it is possible to run chemistry experiments if you have an HDF5 file that has been previously created by a 
driver. The HDF5 driver can do this. See its [readme](./hdf5d/readme.md) for more information    

## Writing a new driver

The drivers here were designed to be pluggable and discoverable. Thus a new driver can be created and simply added and
will be found for use within qischem. If you are writing a new driver to your favorite chemistry program/library then
the driver should derive from BaseDriver class.

A configuration.json file is also needed that names the driver to qischem and specifies the main class that has been
derived from BaseDriver.

The core of the driver should use the chemistry program/library and populate a QMolecule instance with the electronic
structure data.
 
Consulting the existing drivers may be helpful in accomplishing the above.

# Qiskit Chemistry

## Driver for electronic structure previously stored in an HDF5 file

When using a driver, that interfaces to a chemistry program or chemistry library, the electronic structure
information that Qiskit Chemistry obtains and formats into common data structures, for it's subsequent
computation on that molecule, can be saved at that point as an HDF5 file, for later use by this driver.
 
For example, the following input file snippet shows the chemistry program driver for PSI4 being used and the
resulting electronic structure stored in the file molecule.hdf5. The name given here is relative to the folder
where the input file itself resides  
```
&DRIVER
  NAME=PSI4
  HDF5_OUTPUT=molecule.hdf5
&END
```
Once the file has been saved this driver can use it later to do computations on the stored molecule electronic
structure. For example you may wish to do repeated quantum experiments and using this file ensures the exact same
data is used for each test. To use the file in this driver here is a snippet of how to do that
```
&DRIVER
  NAME=HDF5
&END

&HDF5
  HDF5_INPUT=molecule.hdf5
&END
```

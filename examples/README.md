# IBM Quantum Library for Chemistry - Examples

This folder contains a number of example input files that can be loaded and run by the QISChem [GUI](../README.md#gui) 
or run by the [command line](../README.md#command-line) tool.

There are also some example programs and notebooks showing how to use the dictionary equivalent form of
the input file that can be used more effectively programmatically when your goal is to run the content
with a range of different values. For example the [energyplot](energyplot.ipynb) notebook alters the
interatomic distance of a molecule, over a range of values, and uses the results to plot graphs.

## Jupyter Notebook

The folder contains some Jupyter Notebook examples. If you are running directly off a clone of this repository
then on the command line, where you run 'jupyter notebook' to start the server, first change directory
to make this examples folder the current directory. This way the notebooks here will be able to find the
qischem python code in the other folders here (via paths.py which the notebooks include) 

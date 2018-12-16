# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

.PHONY: env doc

# Dependencies need to be installed on the Anaconda virtual environment.
env:
	if test $(findstring QISKitenv, $(shell conda info --envs)); then \
		bash -c "source activate QISKitenv;pip install -r requirements.txt"; \
	else \
		conda create -y -n QISKitenv python=3; \
		bash -c "source activate QISKitenv;pip install -r requirements.txt"; \
	fi;

doc:
	# create Qiskit Chemistry docs
	make -C ../qiskit-chemistry/docs clean
	sphinx-apidoc -f -o ../qiskit-chemistry/docs ../qiskit-chemistry
	make -C ../qiskit-chemistry/docs html
	# create Aqua docs
	make -C docs clean
	sphinx-apidoc -f -o docs .
	make -C docs html

clean: 
	# clean Qiskit Chemistry docs
	make -C ../qiskit-chemistry/docs clean
	# clean Qiskit Aqua docs
	make -C docs clean
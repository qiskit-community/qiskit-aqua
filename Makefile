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
	# create Aqua Chemistry docs
	make -C ../aqua-chemistry/docs clean
	sphinx-apidoc -f -o ../aqua-chemistry/docs ../aqua-chemistry
	make -C ../aqua-chemistry/docs html
	# create Aqua docs
	make -C docs clean
	sphinx-apidoc -f -o docs .
	make -C docs html

clean: 
	# clean Aqua Chemistry docs
	make -C ../aqua-chemistry/docs clean
	# clean Aqua docs
	make -C docs clean
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


.PHONY: lint style test spell

lint:
	pylint -rn --ignore=gauopen qiskit/ml qiskit/aqua qiskit/chemistry qiskit/finance qiskit/optimization test

style:
	pycodestyle --max-line-length=100 --exclude=gauopen qiskit/ml qiskit/aqua qiskit/chemistry qiskit/finance qiskit/optimization test

test:
	stestr run

spell:
	pylint -rn --disable=all --enable=spelling --spelling-dict=en_US --spelling-private-dict-file=.pylintdict --ignore=gauopen qiskit/ml qiskit/aqua qiskit/chemistry qiskit/finance qiskit/optimization test

html:
	make -C docs html

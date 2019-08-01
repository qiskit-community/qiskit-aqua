from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit import transpile


class ExtrapolatedVF(VariationalForm):
    """Variational form that attempts to extrapolate errors for Richardson extrapolation."""

    CONFIGURATION = {
        'name': 'ExtrapolatedVF',
        'description': 'Extrapolated Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ry_schema',
            'type': 'object',
            'properties': {},
            'additionalProperties': False
        }
    }

    def __init__(self, variational_form, pass_manager):
        self.validate(locals())
        super().__init__()
        self.variational_form = variational_form
        self._num_qubits = self.variational_form.num_qubits
        self._num_parameters = self.variational_form.num_parameters
        self._bounds = self.variational_form._bounds
        self._pass_manager = pass_manager

    def construct_circuit(self, parameters, q=None, **transpile_kwargs):
        circ = self.variational_form.construct_circuit(parameters, q=q)
        circ_out = transpile(circ, pass_manager=self._pass_manager, **transpile_kwargs)
        return circ_out

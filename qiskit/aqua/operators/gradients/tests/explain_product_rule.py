# for param, elements in state_qc._parameter_table.items():
#     # Check if we want the gradient for the respective param in the circuit
#     if param not in target_params:
#         continue
#     if param not in params:
#         params.append(param)
#         gates_to_parameters[param] = []
#     for element in elements:
#         # Dict key: param, items: gates using param (paramExpressions tbd)
#         gates_to_parameters[param].append(element[0])
#
# grads = []
# for param in params:
#     grad_param = None
#     for gate in gates_to_parameters[param]:
#         # TODO compute gradient for gate
#         gate_grad = ...
#         if not grad_param:
#             grad_param = gate_grad
#         else:
#             grad_param += gate_grad
#     grads.append(grad_param)
#
# return ListOp(grads)
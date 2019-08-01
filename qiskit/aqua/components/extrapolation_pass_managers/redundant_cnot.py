from qiskit import QuantumRegister
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.propertyset import PropertySet
from qiskit.dagcircuit import DAGCircuit


class RedundantCNOT(TransformationPass):

    def __init__(self, redundant_pairs):
        self.redundant_pairs = redundant_pairs
        if type(self.redundant_pairs) is not int or self.redundant_pairs < 0:
            raise ValueError('Invalid number of redundant pairs given.')
        self.requires = []  # List of passes that requires
        self.preserves = []  # List of passes that preserves
        self.property_set = PropertySet()  # This pass's pointer to the pass manager's property set.
        self._hash = None

    def run(self, dag):
        """
        Run one pass of cx redundant insertion on the circuit

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        if self.redundant_pairs is 0:
            return dag
        cx_runs = dag.collect_runs(["cx"])
        for cx_run in cx_runs:
            # Partition the cx_run into chunks with equal gate arguments
            partition = []
            chunk = []
            for i in range(len(cx_run) - 1):
                chunk.append(cx_run[i])

                qargs0 = cx_run[i].qargs
                qargs1 = cx_run[i + 1].qargs

                if qargs0 != qargs1:
                    partition.append(chunk)
                    chunk = []
            chunk.append(cx_run[-1])
            partition.append(chunk)
            # Simplify each chunk in the partition
            for chunk in partition:
                for n in chunk:
                    redundant_dag = DAGCircuit()
                    qreg = QuantumRegister(2, 'q')
                    cont, targ = qreg[0], qreg[1]
                    new_qargs = [cont, targ]  # TODO: Add barriers
                    redundant_dag.add_qreg(qreg)
                    redundant_dag.apply_operation_back(op=n.op, qargs=new_qargs)  # Insert first op
                    for i in range(self.redundant_pairs):  # Insert redundant pairs
                        redundant_dag.apply_operation_back(op=n.op, qargs=new_qargs)
                        redundant_dag.apply_operation_back(op=n.op, qargs=new_qargs)
                    dag.substitute_node_with_dag(
                        n,
                        input_dag=redundant_dag
                    )
        return dag

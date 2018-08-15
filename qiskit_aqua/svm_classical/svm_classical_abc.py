from abc import ABC, abstractmethod

class SVM_Classical_ABC(ABC):
    def init_args(self, training_dataset, test_dataset, datapoints, print_info, multiclass_alg):
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = list(self.training_dataset.keys())
        self.print_info = print_info
        self.multiclass_alg = multiclass_alg

    @abstractmethod
    def run(self):
        raise NotImplementedError( "Should have implemented this" )



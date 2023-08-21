from __future__ import annotations
from abc import abstractmethod


class EditingLayer():
    def __init__(self, units: int):
        pass

    @abstractmethod
    def add_unit(self):
        pass

    @abstractmethod
    def remove_unit(self):
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def set_input_size(self, size: int) -> int:
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def set_output_size(self, size: int) -> int:
        pass

    @abstractmethod
    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index: int):
        pass


class EditingDense(EditingLayer):
    def __init__(self, units: int = 3, ninputs: int = 3, update_callback = None, name: str = 'Dense'):
        EditingLayer.__init__(self, units)
        self.units = units
        self.ninputs = ninputs
        self.params_init_algorithm = 'random'
        self.activation_function = 'None'
        self.update_callback = update_callback
        self.name = name

    def add_unit(self):
        self.units = self.units + 1
        self.set_output_size(self.units)

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def set_init_params_algorithm(self, algorithm: str):
        self.params_init_algorithm = algorithm

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def get_input_size(self) -> int:
        return self.ninputs

    def set_input_size(self, size: int) -> int:
        self.ninputs = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index: int):
        return list(range(self.units))

class EditingDropout(EditingLayer):
    def __init__(self, units: int = 3, dropout_rate = 0.2, update_callback = None, name: str = 'Dropout') -> None:
        EditingLayer.__init__(self, units)
        self.units = units
        self.dropout_rate = dropout_rate
        self.update_callback = update_callback
        self.name = name

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def set_dropout_rate(self, dropout_rate: float):
        self.dropout_rate = dropout_rate

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]
        

class EditingBinaryCrossEntropy(EditingLayer):
    def __init__(self, units: int = 3, update_callback = None, name: str = 'BinaryCrossEntropy') -> None:
        EditingLayer.__init__(self, units)
        self.units = units
        self.update_callback = update_callback
        self.name = name

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]

class EditingMeanSquaredError(EditingLayer):
    def __init__(self, units: int = 3, update_callback = None, name: str = 'MeanSquaredError') -> None:
        EditingLayer.__init__(self, units)
        self.units = units
        self.update_callback = update_callback
        self.name = name

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]

class EditingBatchNormalization(EditingLayer):
    def __init__(self, units: int, update_callback, name: str = 'BatchNormalization'):
        EditingLayer.__init__(self, units)
        self.update_callback = update_callback
        self.units = units
        self.name = name

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def update(self):
        self.update_callback(self)

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]

class EditingSigmoid(EditingLayer):
    def __init__(self, units: int, update_callback, name: str = 'Sigmoid'):
        self.units = units
        self.name = name
        self.update_callback = update_callback

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]

class EditingReLU(EditingLayer):
    def __init__(self, units: int, update_callback, name: str = 'ReLU'):
        self.units = units
        self.name = name
        self.update_callback = update_callback

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def update(self):
        self.update_callback(self)

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return [previous_node_index]

class EditingSoftmax(EditingLayer):
    def __init__(self, units: int, update_callback, name: str = 'Softmax'):
        self.units = units
        self.name = name
        self.update_callback = update_callback

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def update(self):
        self.update_callback(self)

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

    def get_nodes_indexes_linked_with_previous_node(self, previous_node_index):
        return list(range(self.units))

class EditingInput(EditingLayer):
    def __init__(self, units: int, update_callback, name: str = 'Input'):
        EditingLayer.__init__(self, units)
        self.update_callback = update_callback
        self.units = units
        self.name = name

    def add_unit(self):
        self.units = self.units + 1

    def remove_unit(self):
        if self.units > 1:
            self.units = self.units - 1

    def update(self):
        self.update_callback(self)

    def get_input_size(self) -> int:
        return self.units

    def set_input_size(self, size: int) -> int:
        self.units = size

    def get_output_size(self) -> int:
        return self.units

    def set_output_size(self, size: int) -> int:
        self.units = size

from PyQt5.QtWidgets import QComboBox

from nn_visualizer.application_context.application_context import EditingDense


class ActivationComboBox(QComboBox):
    def __init__(self, editing_dense: EditingDense):
        QComboBox.__init__(self)
        self.editing_dense = editing_dense
        self.items = ['ReLU', 'Sigmoid', 'Softmax', 'Linear', 'None']
        
        for item in self.items:
            self.addItem(item)


        self.setCurrentIndex(self.items.index(self.editing_dense.activation_function))

        self.currentIndexChanged.connect(self.on_index_change)

    def on_index_change(self, index: int):
        activation = self.items[index]
        self.editing_dense.set_activation_function(activation)
        self.editing_dense.update()

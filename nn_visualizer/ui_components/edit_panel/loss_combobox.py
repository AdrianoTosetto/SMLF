from PyQt5.QtWidgets import *


class LossComboBox(QComboBox):
    def __init__(self, parent, on_changed_callback):
        QComboBox.__init__(self)
        self.addItem('Mean Squared Loss')
        self.addItem('Binary Cross Entropy')
        self.addItem('Categorical Cross Entropy')
        self.on_changed_callback = on_changed_callback

        self.activated.connect(self.activated_changed_handler)
        self.currentTextChanged.connect(self.text_changed_handler)
        self.currentIndexChanged.connect(self.index_changed_handler)

    def activated_changed_handler(self, index: int):
        pass

    def text_changed_handler(self, text: str):
        if self.on_changed_callback is not None: self.on_changed_callback(text)

    def index_changed_handler(self, index: int):
        pass

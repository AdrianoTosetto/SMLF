from typing import Callable

from PyQt5.QtWidgets import *

from nn_visualizer.ui_components.customized.label.item_label import ItemLabel


class ComponentNameContainer(QWidget):
    def __init__(self, parent: QWidget, name_change_callback: Callable[[str], None], default_text = '') -> None:
        QWidget.__init__(self, parent)
        self.name_change_callback = name_change_callback
        self.build_layout(default_text)

    def build_layout(self, default_text):
        root_layout = QHBoxLayout()

        label_name = ItemLabel('Name')
        edit_name = QLineEdit()
        edit_name.setText(default_text)

        root_layout.addWidget(label_name)
        root_layout.addWidget(edit_name)

        self.setLayout(root_layout)

        edit_name.textChanged.connect(self.name_changed_handler)

    def name_changed_handler(self, text):
        print(text)
        self.text = text
        self.name_change_callback(text)

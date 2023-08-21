import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import *

from nn_visualizer.application_context.application_context import (
    EditingDense,
    EditingLayer,
)
from nn_visualizer.ui_components.customized.button.smlf_icon_button import SMLFIconButton


class AddUnitButton(SMLFIconButton):
    def __init__(self, handler = lambda _: _):
        SMLFIconButton.__init__(self, icon=qta.icon('mdi.plus', color='white'), width=28, height=28, click_handler=handler)

class RemoveUnitButton(SMLFIconButton):
    def __init__(self, handler = lambda _: _):
        SMLFIconButton.__init__(self, icon=qta.icon('mdi.minus', color='white'), width=28, height=28, click_handler=handler)

class UnitsNumberEditor(QWidget):
    def __init__(self, editing_dense: EditingLayer) -> None:
        QWidget.__init__(self)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.editing_dense = editing_dense
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        add_neuron_button = AddUnitButton(handler=self.add_unit_button_handler)
        remove_neuron_button = RemoveUnitButton(handler=self.remove_unit_button_handler)
        root_layout.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum
        )

        root_layout.addWidget(add_neuron_button)
        root_layout.addWidget(remove_neuron_button)
        self.setLayout(root_layout)

    def add_unit_button_handler(self, event: QMouseEvent):
        self.editing_dense.add_unit()
        self.editing_dense.update()

    def remove_unit_button_handler(self, event: QMouseEvent):
        self.editing_dense.remove_unit()
        self.editing_dense.update()

    def size(self):
        return QSize(64, 30)

    def sizeHint(self):
        return QSize(64, 30)

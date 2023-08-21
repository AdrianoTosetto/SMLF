import qtawesome as qta
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.ui_components.customized.button.smlf_icon_button import SMLFIconButton


class NextStepButton(SMLFIconButton):
    def __init__(self, handler = lambda _ : _):
        SMLFIconButton.__init__(
            self, icon=qta.icon('mdi.skip-next', color='white'),
            click_handler=handler,
            width=32,
            height=32
        )

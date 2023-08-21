import qtawesome as qta
from PyQt5.QtGui import QMouseEvent

from nn_visualizer.ui_components.customized.button.smlf_icon_button import SMLFIconButton


class ResetButton(SMLFIconButton):
    def __init__(self, handler = lambda _: _):
        SMLFIconButton.__init__(
            self, icon=qta.icon('mdi.replay', color='white'),
            width=36,
            height=36,
            click_handler=self.click_handler
        )
        self.handler = handler

    def click_handler(self, event: QMouseEvent):
        self.handler(2)

import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *


def clickable(widget):
    class Filter(QObject):

        clicked = pyqtSignal()
        
        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.Enter:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()

                        return True
            
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)

    return filter.clicked

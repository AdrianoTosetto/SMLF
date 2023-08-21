import json

import PyQt5
import qtawesome as qta
from PyQt5.QtCore import QMimeData, QPoint, Qt
from PyQt5.QtGui import QCursor, QDrag, QPixmap
from PyQt5.QtWidgets import QLabel, QPushButton


class DragButton(QPushButton):

    def __init__(self, parent, text: str, drag_payload: dict = dict({})):
        QPushButton.__init__(self, parent)
        self.setEnabled(True)
        self.setText(text)
        self.drag_payload = drag_payload

    def mouseMoveEvent(self, e):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(json.dumps(self.drag_payload))

        pixmap = QPixmap(self.size())
        self.render(pixmap)
        drag.setPixmap(pixmap)

        drag.setMimeData(mime)
        dropAction = drag.exec_(Qt.MoveAction)

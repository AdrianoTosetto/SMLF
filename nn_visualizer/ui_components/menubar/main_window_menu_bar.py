from typing import Callable

from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt

from nn_visualizer.application_context.application_context import ApplicationState


class MainWindowMenuBar(QMenuBar):

    def __init__(self, parent: QWidget = None, open_file_callback = None, switch_view_mode_callback = None):
        print(parent)
        QMenu.__init__(self, parent)
        self._parent = parent
        self.open_file_callback = open_file_callback
        # self.setFixedHeight(50)
        effect = QGraphicsDropShadowEffect(
        offset=QPoint(3, 3), blurRadius=25, color=QColor("#e0e0e0"))
    
        self.setGraphicsEffect(effect)
        self.setStyleSheet("{{background-color: red;}}")
        self.setNativeMenuBar(False)
        actionFile = self.addMenu("File")
        actionFile.addAction("New model")
        actionFile.addSeparator()
        actionFile.addAction(self.build_open_model_action())
        actionFile.addMenu("Open Recent")
        actionView = self.addMenu("View")
        mode_menu = actionView.addMenu("Mode")
        mode_menu.addAction(self.build_editing_action())
        mode_menu.addAction(self.build_training_action())

        self.switch_view_mode_callback: Callable[[ApplicationState], None] = switch_view_mode_callback

        self.show()

    def open_file(self):
        filter = "json(*.json)"
        path, _ = QFileDialog.getOpenFileName(self._parent, "Open Model")

        self.open_file_callback(path)

    def build_new_model_action(self): pass

    def build_editing_action(self) -> QAction:
        editing_action = QAction("Edit", self._parent)
        editing_action.triggered.connect(self.mode_menu_editing_handler)

        return editing_action

    def build_training_action(self) -> QAction:
        training_action = QAction("Training", self._parent)
        training_action.triggered.connect(self.mode_menu_training_handler)

        return training_action

    def build_open_model_action(self) -> QAction:
        new_model_action = QAction("Open Model", self._parent)
        new_model_action.triggered.connect(self.open_file)

        return new_model_action

    def mode_menu_editing_handler(self, ):
        self.switch_view_mode_callback(ApplicationState.EDITING)

    def mode_menu_training_handler(self, ):
        self.switch_view_mode_callback(ApplicationState.IDLE_TRAINING)

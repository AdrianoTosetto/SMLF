import os
import sys
from functools import reduce

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from api.sequential.design_patterns.observer.observable import Event
from api.sequential.pipeline.pipeline_mode import PipelineMode
from nn_visualizer.application_context.application_context import ApplicationContext
from nn_visualizer.ui_components.theme.theme import Colors


class ConsolePath():
    def __init__(self, path: str):
        self.path = path

    def go_to_parent(self):
        self.path = reduce(str.__add__, self.path.split('/')[0:-1], '')

class Console(QTextEdit):
    def __init__(self, application_context: ApplicationContext, handle_load_callback = None):
        QTextEdit.__init__(self, )
        self.setObjectName('Console')
        self.setFixedHeight(200)
        self.application_context = application_context

        self.setStyleSheet('QTextEdit {{color: white; font-family: Courier;}} #Console {{ background-color: {color}; }}'.format(color=Colors.dark_blue))
        self.textChanged.connect(self.text_changed_handler)
        self.current_line = ''
        self.path = os.getcwd()
        self.add_new_line_no_break()
        self.handle_load_callback = handle_load_callback
        self.current_line_index = 0


    def get_console_line_begin(self):
        return '{path}:/> '.format(path=self.path)

    def text_changed_handler(self):
        text = self.toPlainText()

    def add_new_line_no_break(self):
        self.append(self.get_console_line_begin())
        self.current_line = ''
        self.current_line_index = 0

    def add_new_line(self):
        self.append('{line}'.format(line=self.get_console_line_begin()))
        self.current_line = ''
        self.current_line_index = 0

    def keyPressEvent(self, event: QKeyEvent):
        self.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        if event.key() == Qt.Key_Return:
            output = self.handle_enter()
            self.add_content(output)
            self.add_new_line_no_break()
        else:
            if event.key() != Qt.Key_Backspace:
                self.current_line += event.text()
                self.current_line_index += 1
                super().keyPressEvent(event)
            else:
                if self.current_line_index > 0:
                    self.current_line_index -= 1
                    self.current_line = self.current_line[0:-1]
                    super().keyPressEvent(event)

    def handle_enter(self):
        command_line = self.current_line
        tokens = command_line.removeprefix(' ').removesuffix(' ').split(' ')
        command = tokens[0]

        if command == 'load':
            script = tokens[1]
            full_path = '{path}/{script}'.format(path=self.path, script=script)
            import importlib.machinery
            import importlib.util
            import numpy as np

            script_name = tokens[1]
            loader = importlib.machinery.SourceFileLoader(
                'dataset_module',
                '{path}/{script}'.format(path=self.path, script=script_name)
            )

            spec = importlib.util.spec_from_loader('dataset_module', loader )
            dataset_module = importlib.util.module_from_spec(spec)
            loader.exec_module( dataset_module )
            (dataset, targets, validation_dataset, validation_targets) = dataset_module.get()
            self.application_context.editing_layers[0].set_input_size(dataset.shape[1])
            self.application_context.editing_layers[0].update()
            self.handle_load_callback(dataset, targets, validation_dataset, validation_targets)

            return 'Loaded dataset, shape = {shape}'.format(shape=dataset.shape)

        if command == 'cd':
            arg = tokens[1]
            if arg == '..':
                import re
                self.path = reduce(str.__add__, re.split('([/])', self.path)[0:-2], '')
            else:
                self.path += '/{dir}'.format(dir=arg)

            return ''

        if command == 'clear':
            self.setText('')
        
            return ''

    def add_content(self, text):
        self.moveCursor(QTextCursor.End)
        self.append(text)

        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def pipeline_update_handler(self, event: Event):
        print(event.payload)

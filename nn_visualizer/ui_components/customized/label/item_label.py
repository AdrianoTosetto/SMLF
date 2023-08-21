import uuid

from PyQt5 import Qt, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.ui_components.theme.theme import TextSize
from nn_visualizer.ui_components.theme.utils import json_style_to_css


font_sizes = dict({
    'xs': 8,
    'sm': 12,
    'md': 16,
    'lg': 20,
    'xs': 24,
    'xl': 28,
})
    
class ItemLabel(QLabel):

    def __init__(self, text: str = '', size: str = 'md'):
        QLabel.__init__(self, text)
        self.setObjectName('ItemLabel')
        self.setFixedHeight(32)
        self.setStyleSheet("#ItemLabel {{ {style} }}".format(style=json_style_to_css(self.default_style(size))))

    def default_style(self, size: str) -> dict:
        font_size = font_sizes[size]
        return dict({
            'font-size': '{size}px'.format(size=font_size),
            'color': '#333333',
            'background-color': 'transparent',
        })

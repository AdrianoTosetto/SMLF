import sys
import typing
from functools import reduce
from itertools import accumulate, starmap

import qtawesome as qta
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget


dark_blue = '#0C2D48'
blue = '#2E8BC0'
baby_blue = '#B1D4E0'
midnight_blue = '#145DA0'

class SMLFIconButton(QPushButton):
    def __init__(self, icon: QIcon = None, click_handler = lambda event: None, width: int = 64, height: int = 64):
        QPushButton.__init__(self)
        self.setIcon(icon)
        self.setFixedHeight(height)
        self.setFixedWidth(width)
        self.setIconSize(QtCore.QSize(int(width*0.8), int(height*0.8)))
        self.setStyleSheet("background-color: {hex}; border: none; border-radius: 4px;".format(hex = dark_blue))
        self.click_handler = click_handler

    def default_style(self) -> dict:
        return dict({
            'background-color': dark_blue,
            'border': 'none',
            'border-radius': '4px',
        })

    def set_style(self, style: dict) -> None:
        self.setStyleSheet(self.json_style_to_css(style))

    def json_style_to_css(self, json: dict) -> str:

        kv_to_string = lambda key, value: '{key}: {value}'.format(key=key, value=value)

        style_props = list(starmap(kv_to_string, zip(json.keys(), json.values())))

        return reduce(lambda acc, curr: '{acc}{curr};'.format(acc=acc, curr=curr), style_props, '')
        

    def mousePressEvent(self, e: QMouseEvent) -> None:
        return self.click_handler(e)

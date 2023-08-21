import math

from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.geometry.point import Point


class LineSegment():
    def __init__(self, a: Point, b: Point) -> None:
        self.a = a
        self.b = b

    def length(self) -> float:
        return math.sqrt(math.pow(self.a.x - self.b.x, 2) + math.pow(self.a.y - self.b.y, 2))

    def unit_vector(self):
        point_vector = self.b - self.a

        return point_vector * (1.0 / self.length())

    def to_QLine(self) -> QLine:
        a = QPoint(int(self.a.x), int(self.a.y))
        b = QPoint(int(self.b.x), int(self.b.y))

        return QLine(a,b)

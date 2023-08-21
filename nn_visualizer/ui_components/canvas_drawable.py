from abc import ABC, abstractmethod

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon
from nn_visualizer.geometry.utils import trace_path, tracing_current_origin


class CanvasDrawable():
    def __init__(self):
        pass

    @abstractmethod
    def draw(painter: QPainter):
        pass

    @abstractmethod
    def width(self) -> float:
        pass

    @abstractmethod
    def height(self) -> float:
        pass

    '''
        padding setters
    '''

    @abstractmethod
    def set_padding_left(self, padding: int) -> None:
        pass
    @abstractmethod
    def set_padding_right(self, padding: int) -> None:
        pass

    @abstractmethod
    def set_padding_top(self, padding: int) -> None:
        pass

    @abstractmethod
    def set_vertical_padding(self, padding: tuple[int, int]) -> None:
        pass

    @abstractmethod
    def set_horizontal_padding(self, padding: tuple[int, int]) -> None:
        pass

    @abstractmethod
    def set_padding_bottom(self, padding: int) -> None:
        pass

    @abstractmethod
    def set_padding(self, padding: tuple[int, int, int, int]) -> None:
        pass

    
    '''
        padding getters
    '''

    @abstractmethod
    def get_padding_left(self) -> int:
        pass

    @abstractmethod
    def get_padding_right(self) -> int:
        pass

    @abstractmethod
    def get_padding_top(self) -> int:
        pass

    @abstractmethod
    def get_padding_bottom(self) -> int:
        pass
    

    @abstractmethod
    def get_vertical_padding(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_horizontal_padding(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_padding(self, padding) -> tuple[int, int, int, int]:
        pass
    
    @abstractmethod
    def origin(self):
        pass

    @abstractmethod
    def set_origin(self, x, y):
        pass

    @abstractmethod
    def get_center_x() -> float:
        pass

    @abstractmethod
    def get_center_y() -> float:
        pass

    @abstractmethod
    def get_shape(self,) -> Polygon: 
        pass

    @abstractmethod
    def on_mouse_press(self, event: QMouseEvent):
        pass

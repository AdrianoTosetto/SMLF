from __future__ import annotations
import math


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def distance(self, other):
        return math.sqrt(math.pow(self.x - other.x) + math.pow(self.y - other.y))

    def __add__(self, other: Point) -> Point:
        x = self.x + other.x
        y = self.y + other.y

        return Point(x, y)

    def __sub__(self, other: Point) -> Point:
        dx = self.x - other.x
        dy = self.y - other.y

        return Point(dx, dy)

    def __mul__(self, scalar: float) -> Point:
        if isinstance(scalar, int) or isinstance(scalar, float):
            return Point(scalar * self.x, scalar * self.y)
        else:
            raise Exception('Invalid parameter for mul')

    def __rmul__(self, scalar: float) -> Point:
        if isinstance(scalar, int) or isinstance(scalar, float):
            return Point(scalar * self.x, scalar * self.y)
        else:
            raise Exception('Invalid parameter for mul')

    def __truediv__(self, scalar: float) -> Point:
        factor: float = 1.0 / scalar
        if isinstance(scalar, int) or isinstance(scalar, float):
            return self.__mul__(factor)
        else:
            raise Exception('Invalid parameter for div')

    def __str__(self) -> str:
        return '({x: .2f},{y: .2f})'.format(x = self.x, y = self.y)

    def copy(self):
        return Point(self.x, self.y)

from __future__ import annotations
import copy
import math
from enum import Enum
from functools import reduce

from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point


class Polygon():
    def __init__(self, edges: list[LineSegment]) -> None:
        self.edges = edges

    def length(self) -> float:
        return reduce(lambda acc, curr: acc + curr.length(), self.edges, 0.0)

    @classmethod
    def from_vertices(cls, vertices: list[Point]) -> Polygon:
        shifted_vertices = copy.deepcopy(vertices)
        first = shifted_vertices[0]
        del shifted_vertices[0]
        shifted_vertices.append(first)

        edges = list(map(lambda points: copy.deepcopy(LineSegment(points[0], points[1])), zip(vertices, shifted_vertices)))

        return cls(edges)

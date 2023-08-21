import copy
from enum import Enum
from functools import reduce
from math import pow, sqrt

from nn_visualizer.geometry.circle import Circle
from nn_visualizer.geometry.line_segment import LineSegment
from nn_visualizer.geometry.point import Point
from nn_visualizer.geometry.polygon import Polygon


tolerance = 1e-07

class TripletPointOrientation(Enum):
    Collinear = 0
    Clockwise = 1
    CounterClockwise = 2

def is_zero(number: float):
    return -tolerance <= number <= tolerance

def distance(a: Point, b: Point):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))

def orientation(a: Point, b: Point, c: Point) -> TripletPointOrientation:
    tmp = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)

    if (is_zero(tmp)): return TripletPointOrientation.Collinear

    if (tmp > 0): return TripletPointOrientation.Clockwise

    if (tmp < 0): return TripletPointOrientation.CounterClockwise

def line_segments_intersects(lhs: LineSegment, rhs: LineSegment) -> bool:
    o1 = orientation(lhs.a, lhs.b, rhs.a)
    o2 = orientation(lhs.a, lhs.b, rhs.b)
    o3 = orientation(rhs.a, rhs.b, lhs.a)
    o4 = orientation(rhs.a, rhs.b, lhs.b)

    if (o1 != o2 and o3 != o4): return True

    if (o1 == TripletPointOrientation.Collinear and point_lies_on_line_segment(lhs, rhs.a)):
        return True

    if (o2 == TripletPointOrientation.Collinear and point_lies_on_line_segment(lhs, rhs.b)):
        return True

    if (o3 == TripletPointOrientation.Collinear and point_lies_on_line_segment(rhs, lhs.a)):
        return True

    if (o4 == TripletPointOrientation.Collinear and point_lies_on_line_segment(rhs, lhs.b)):
        return True

    return False

def point_lies_on_line_segment(line_segment: LineSegment, point: Point):
    distance_ab = distance(line_segment.a, line_segment.b)

    distance_aP = distance(line_segment.a, point)
    distance_Pb = distance(point, line_segment.b)

    return abs(distance_aP + distance_Pb - distance_ab) <= tolerance

def point_lies_on_polygon(polygon: Polygon, point: Point) -> bool:
    extended_line_segment = LineSegment(point, Point(point.x + 1000, point.y))
    count = 0

    for line_segment in polygon.edges:
        if line_segments_intersects(line_segment, extended_line_segment):
            count += 1

    return count & 1

def point_lies_on_circle(circle: Circle, point: Point) -> bool:
    (x, y) = circle.center()
    circle_center = Point(x, y)
    radius = circle.radius

    return distance(circle_center, point) <= radius

def tracing_current_origin(polygon: Polygon, completed: float) -> Point:
    polygon_length = reduce(lambda acc, curr: acc + curr.length(), polygon.edges, 0.0)
    completed_path_length = completed * polygon_length

    traced_path_length = 0.0
    point = polygon.edges[0].a.copy()
    index = 0

    for line_segment in polygon.edges:
        segment_length = line_segment.length()

        if traced_path_length + segment_length < completed_path_length:
            traced_path_length += segment_length
        else:
            remaining_path_length = completed_path_length - traced_path_length
            point = line_segment.a + (line_segment.unit_vector() * remaining_path_length)
            break

        index+=1

    return (point, index)

def trace_path(polygon: Polygon, origin: Point, length: float, index: int) -> LineSegment:
    polygon_path = copy.deepcopy(polygon.edges)
    head = polygon_path[0:index]
    del polygon_path[0:index]
    polygon_path.extend(head)
    polygon_path[0].a = copy.deepcopy(origin)

    remaining_length = length
    traced_path: list[LineSegment] = []

    for line_segment in polygon_path:
        line_segment_length = line_segment.length()

        if (remaining_length - line_segment_length) > 0:
            remaining_length -= line_segment_length
            traced_path.append(line_segment)

        else:
            end_point = line_segment.unit_vector().__mul__(remaining_length)
            line_segment.b = end_point + line_segment.a
            traced_path.append(line_segment)
            break


    return traced_path

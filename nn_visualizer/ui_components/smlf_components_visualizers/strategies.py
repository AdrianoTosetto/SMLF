import PyQt5
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QPainter, QPen

from nn_visualizer.ui_components.smlf_components_visualizers.layer_layout_data import LayerVisualizerLayoutData


def line_width_from_ndarray(ndarray: np.ndarray, value: float):
    from scipy.interpolate import interp1d

    max = np.max(ndarray)
    min = np.min(ndarray)

    line_width_range = [1, 5]
    ndarray_range = [min, max]
    
    width_interp = interp1d(ndarray_range, line_width_range)

    return width_interp(value)

def line_color_from_derivative(value: float):
    from scipy.interpolate import interp1d
    derivatives_range = [0, .1]
    colors_range = [200, 0]

    if value > derivatives_range[1]:
        return 0

    color_interp = interp1d([0, 5], [255, 0])

    return color_interp(value)

def line_color_from_derivative(value):
    value = np.abs(value)

    minimum, maximum = 0, .1
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (np.log(value+1)-minimum) / (maximum - minimum)

    b = int(max(0, min(255, 255*(1 - ratio))))
    r = int(max(0, min(255, 255*(ratio - 1))))
    g = 255 - b - r
    return r, g, b

def draw_single_connected_backward_connection(
        painter: QPainter,
        layer_layout_data: LayerVisualizerLayoutData,
        index: int,
    ):

    units_origins = layer_layout_data.get_units_origins()
    unit_origin = units_origins[index]
    unit_radius = layer_layout_data.get_unit_radius()

    output_origins = layer_layout_data.get_output_units_origins()
    output_origin = output_origins[index]
    derivatives = layer_layout_data.get_backpropagation_derivatives()
    line_width = 1
    line_color = Qt.green

    if derivatives is not None:
        derivative = derivatives[index]
        line_width = line_width_from_ndarray(derivatives, derivatives[index])
        # line_color = QColor(gray_scale, gray_scale, gray_scale)
        # gray_scale = int(line_color_from_derivative(np.abs(derivative)))
        r, g, b = line_color_from_derivative(np.abs(derivative))
        line_color = QColor(r, g, b)

    painter.setPen(QPen(line_color, line_width))
    a = QPoint(int(unit_origin[0] + unit_radius), int(unit_origin[1]))
    b = QPoint(int(output_origin[0]), int(output_origin[1]))
    painter.drawLine(a, b)

def draw_fully_connected_backward_connections(
        painter: QPainter,
        layer_layout_data: LayerVisualizerLayoutData,
        index: int
    ):

    units_origins = layer_layout_data.get_units_origins()
    unit_origin = units_origins[index]
    unit_radius = layer_layout_data.get_unit_radius()

    output_origins = layer_layout_data.get_output_units_origins()
    derivatives = layer_layout_data.get_backpropagation_derivatives()

    line_width = 1
    line_color = Qt.green
    fill_color = Qt.white

    for unit_index, output_origin in enumerate(output_origins):
        if derivatives is not None:
            derivative = derivatives[unit_index][index]
            line_width = line_width_from_ndarray(derivatives, derivative)
            # gray_scale = int(line_color_from_derivative(np.abs(derivative)))
            # line_color = QColor(gray_scale, gray_scale, gray_scale)
            r, g, b = line_color_from_derivative(np.abs(derivative))
            line_color = QColor(r, g, b)

        a = QPoint(int(unit_origin[0] + unit_radius), int(unit_origin[1]))
        b = QPoint(int(output_origin[0]), int(output_origin[1]))
        painter.setPen(QPen(line_color, line_width))
        painter.drawLine(a, b)

def draw_single_connected_forward_connection(
        painter: QPainter,
        layer_layout_data: LayerVisualizerLayoutData,
        index: int,
    ):

    units_origins = layer_layout_data.get_units_origins()
    unit_origin = units_origins[index]
    unit_radius = layer_layout_data.get_unit_radius()

    input_origins = layer_layout_data.get_input_units_origins()
    input_origin = input_origins[index]
    derivatives = layer_layout_data.get_backpropagation_derivatives()
    line_width = 1

    painter.setPen(QPen(Qt.green, line_width))
    a = QPoint(int(unit_origin[0] - unit_radius), int(unit_origin[1]))
    b = QPoint(int(input_origin[0]), int(input_origin[1]))
    painter.drawLine(a, b)

def draw_fully_connected_forward_connections(
        painter: QPainter,
        layer_layout_data: LayerVisualizerLayoutData,
        index: int
    ):

    units_origins = layer_layout_data.get_units_origins()
    unit_origin = units_origins[index]
    unit_radius = layer_layout_data.get_unit_radius()

    input_origins = layer_layout_data.get_input_units_origins()
    derivatives = layer_layout_data.get_backpropagation_derivatives()

    line_width = 1

    for unit_index, input_origin in enumerate(input_origins):
        a = QPoint(int(unit_origin[0] - unit_radius), int(unit_origin[1]))
        b = QPoint(int(input_origin[0]), int(input_origin[1]))

        painter.setPen(QPen(Qt.green, line_width))
        painter.drawLine(a, b)

# def draw_

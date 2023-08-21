import numpy as np
import qtawesome as qta
from PyQt5 import Qt, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import *

from nn_visualizer.application_context.editing_layers import EditingBatchNormalization
from nn_visualizer.ui_components.components_editors.editor_container import EditorContainer


class BatchNormalizationEditor(EditorContainer):
    def __init__(self, editing_batch_normalization: EditingBatchNormalization):
        EditorContainer.__init__(self, editing_batch_normalization, [])

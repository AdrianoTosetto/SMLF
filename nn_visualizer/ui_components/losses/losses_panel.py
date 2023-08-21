import numpy
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from pyqtgraph import PlotWidget, plot

from nn_visualizer.ui_components.customized.label.item_label import ItemLabel


class TrainLoss(pg.PlotWidget):
    def __init__(self):
        pg.PlotWidget.__init__(self)
        self.graphWidget = pg.PlotWidget()
        self.imv = pg.ImageView()

class TestLoss(pg.PlotWidget):
    def __init__(self):
        pg.PlotWidget.__init__(self)
        self.graphWidget = pg.PlotWidget()

class LossesPanel(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

        plot_widget = pg.PlotWidget(background='#145DA0', parent=self)
        plot_widget2 = pg.PlotWidget(background='#145DA0', parent=self)
        self.plot = plot_widget.plot(np.arange(0), np.zeros(0), pen='w')
        self.validation_plot = plot_widget2.plot(np.arange(0), np.zeros(0))

        vb = self.plot.getViewBox()       
        vb.setAspectLocked(lock=False)
        vb.setAutoVisible(y=1.0)
        vb.enableAutoRange(axis='y', enable=True)
        vb.enableAutoRange(axis='x', enable=True)
        self.setFixedHeight(300)

        vb = self.validation_plot.getViewBox()
        vb.setAspectLocked(lock=False)
        vb.setAutoVisible(y=1.0)
        vb.enableAutoRange(axis='y', enable=True)
        self.setFixedHeight(300)
        self.current_train_loss = 'X'
        self.current_validation_loss = 'X'

        root_layout = QVBoxLayout()
        losses_titles_layout = QHBoxLayout()
        self.train_loss_label = ItemLabel('Train Loss: 0.00')
        self.validation_loss_label = ItemLabel('Validation Loss: 0.00')
        losses_titles_layout.addWidget(self.train_loss_label)
        losses_titles_layout.addWidget(self.validation_loss_label)

        plots_layout = QHBoxLayout()
        plots_layout.addWidget(plot_widget)
        plots_layout.addWidget(plot_widget2)

        root_layout.addLayout(losses_titles_layout)
        root_layout.addLayout(plots_layout)

        # root_layout.addWidget(plot_widget)
        # root_layout.addWidget(plot_widget2)
        self.setLayout(root_layout)

    def _get_train_loss_label(self, loss):
        return 'Train Loss: {:.2f}'.format(loss)

    def _get_validation_loss_label(self, loss):
        return 'Validation Loss: {:.2f}'.format(loss)

    def update_train_loss_graph(self, epochs, losses, validation_losses):
        train_loss = losses[-1]
        validation_loss = validation_losses[-1]
        self.train_loss_label.setText(self._get_train_loss_label(train_loss))
        self.validation_loss_label.setText(self._get_validation_loss_label(validation_loss))
        self.plot.setData(epochs, losses)
        self.validation_plot.setData(epochs, validation_losses)
        self.update()

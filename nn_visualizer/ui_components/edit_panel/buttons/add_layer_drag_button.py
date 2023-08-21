from nn_visualizer.ui_components.customized.drag_drop.smlf_drag_button import DragButton
from nn_visualizer.ui_components.theme.theme import Colors
from nn_visualizer.ui_components.theme.utils import json_style_to_css


class AddLayerDragButton(DragButton):
    def __init__(self, parent, layer_name: str, drag_payload: dict = dict({})):
        DragButton.__init__(self, parent, layer_name, drag_payload=drag_payload)
        style_tr = json_style_to_css({
           'border': 'none',
           'border-radius': '8px',
           'background': Colors.baby_blue,
           'padding': '24px',
           'color': 'white',
           'font-size': '18px',
        })

        self.setStyleSheet(style_tr)

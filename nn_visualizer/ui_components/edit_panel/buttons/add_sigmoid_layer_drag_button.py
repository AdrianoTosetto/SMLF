from nn_visualizer.ui_components.edit_panel.buttons.add_layer_drag_button import AddLayerDragButton


class AddSigmoidDragButton(AddLayerDragButton):
    def __init__(self, parent):
        drag_payload = dict({
            'layer_name': 'Sigmoid',
            'units': '3',
        })

        AddLayerDragButton.__init__(self, parent, 'Sigmoid', drag_payload)

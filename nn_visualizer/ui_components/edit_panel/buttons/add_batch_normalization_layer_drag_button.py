from nn_visualizer.ui_components.edit_panel.buttons.add_layer_drag_button import AddLayerDragButton


class AddBatchNormalizationDragButton(AddLayerDragButton):
    def __init__(self, parent, *args, **kwargs):
        drag_payload = dict({
            'layer_name': 'BatchNormalization',
            'units': '3',
        })

        AddLayerDragButton.__init__(self, parent, 'Batch Normalization', drag_payload)

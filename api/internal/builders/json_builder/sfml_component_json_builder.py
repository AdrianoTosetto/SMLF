
def json_router():
    import api.layers.dense as dense1
    from api.layers.activation.sigmoid import Sigmoid
    from api.layers.activation.relu import ReLU
    from api.layers.activation.softmax import Softmax, SoftmaxCategoricalCrossEntropyOptimization
    from api.losses.regression.mean_squared_loss import MeanSquareError
    from api.layers.inv_dropout import Dropout

    return dict({
        'Dense': dense1.Dense.from_json,
        'Sigmoid': Sigmoid.from_json,
        'ReLU': ReLU.from_json,
        'Softmax': Softmax.from_json,
        'SoftmaxCategoricalCrossEntropyOptimization': SoftmaxCategoricalCrossEntropyOptimization.from_json,
        'Dropout': Dropout.from_json,
        'MeanSquaredError': MeanSquareError.from_json
    })

class SFMLComponentJsonBuilder():
    @classmethod
    def build(cls, json_object: dict):
        component_name = json_object['component_name']
        _builder = json_router()[component_name]

        if _builder is None:
            raise Exception('Invalid json object')
        else:
            return _builder(json_object)

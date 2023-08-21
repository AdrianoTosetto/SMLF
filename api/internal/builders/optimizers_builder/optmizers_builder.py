def _sgd_builder(learning_rate):
    from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent

    return StochasticGradientDescent(learning_rate)

def _adagrad_builder(learning_rate: float):
    from api.optimizers.adagrad_optimizer import AdagradOptmizer

    return AdagradOptmizer(learning_rate)


def name_builder_router():

    return dict({
        'adagrad': _adagrad_builder,
        'sgd': _sgd_builder, 
    })

class OptimizerBuilder():

    @staticmethod
    def build_from_name(cls, name, learning_rate=0.01):
        name_builder_router = name_builder_router()

        builder = name_builder_router[name]

        if builder is None:
            raise Exception('Invalid optimizer name')

        return builder(learning_rate)

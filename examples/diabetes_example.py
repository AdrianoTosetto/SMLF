import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from api.batch_dense_normalization import BatchDenseNormalization
from api.layers.activation.sigmoid import Sigmoid
from api.layers.activation.softmax import Softmax
from api.layers.dense import Dense
from api.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from api.sequential.model import SequentialModel



data = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',delimiter =',')


print(data.shape)

X_data = data[:,0:8]
print('X_data:',np.shape(X_data))
Y_data = data[:,8].reshape(-1, 1)
print('Y_data:',np.shape(Y_data))

print(X_data)
X_data = preprocessing.normalize(X_data)
print(X_data)


model = SequentialModel(
    layers=[
        Dense(ninputs=8, noutputs=4, init_algorithm='uniform', activation='relu'),
        Dense(ninputs=4, noutputs=8, init_algorithm='uniform', activation='sigmoid'),
        Dense(ninputs=8, noutputs=1, init_algorithm='uniform'),
        Softmax(1)
    ],
    optimizer=StochasticGradientDescent(learning_rate=0.1), loss='categorical_cross_entropy'
)

model.fit(X_data, Y_data, epochs=10000, batch_size=500)

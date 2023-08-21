import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.conv2d_layer import Conv2DLayer

layer = Conv2DLayer(kernel_size=(3,3), input_size=(32, 32, 3), nkernels=3, padding='same')

from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD
import numpy as np
import os

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalWithChannels:

    @staticmethod
    def train(data, size, channels):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)
        model = Sequential([
            Conv2D(filters=5,
                   kernel_size=(5, 5),
                   activation='relu',
                   data_format='channels_first',
                   input_shape=(channels, size, size)),
            Conv2D(filters=1,
                   kernel_size=(5, 5),
                   activation='relu',
                   data_format='channels_first')
        ])

        model.compile(
            optimizer=SGD(lr=0.001),
            loss='mse'
        )

        model.fit(x, y, epochs=50)
        model.save('conv_chan_model.h5')

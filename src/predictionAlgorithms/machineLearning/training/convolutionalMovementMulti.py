from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD
import numpy as np
import os

from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.callbacks import Callbacks
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalWithChannelsMovement:

    @staticmethod
    def train(data, size, channels, validation_data):
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
                   kernel_size=(3, 3),
                   activation='relu',
                   data_format='channels_first')
        ])

        callback = Callbacks()
        callback\
            .set_algorithm(ConvolutionalChannelsMovementAlgorithm())\
            .set_validation_data(validation_data)\
            .set_validation_frequency(1)

        model.compile(
            optimizer=SGD(lr=0.001),
            loss='mse'
        )

        model.fit(x, y, epochs=20, callbacks=[callback])
        model.save('conv_chan_movement_model.h5')

import tensorflow
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import SGD
import numpy as np
import os
from keras import backend as K

from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.callbacks import Callbacks
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalWithChannelsMovement:

    @staticmethod
    def train(data, size, channels, validation_data):
        x, y = data

        model = Sequential()
        model.add(
            Conv2D(
                   filters=4,
                   kernel_size=(7, 7),
                   activation='relu',
                   input_shape=(channels, size, size),
                   data_format='channels_first')
        )
        model.add(
            Conv2DTranspose(filters=1,
                            kernel_size=(7, 7),
                            activation='relu',
                            data_format='channels_first')
        )




        model.compile(
            optimizer=SGD(lr=0.01),
            loss='mse'
        )

        callback = Callbacks()
        callback \
            .set_algorithm(ConvolutionalChannelsMovementAlgorithm(model=model)) \
            .set_validation_data(validation_data) \
            .set_size(size) \
            .set_validation_frequency(1)

        model.fit(x, y, epochs=10, shuffle=True, callbacks=[callback])
        model.save('conv_chan_movement_model_4.h5')

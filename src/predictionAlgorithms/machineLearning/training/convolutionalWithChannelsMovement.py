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
    def train(data, size, channels, validation_data, loader, step=1):
        # x, y = data
        kernel = 10

        model = Sequential()
        model.add(
            Conv2D(
                   filters=channels,
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
            optimizer=SGD(lr=0.005),
            loss='mse'
        )

        callback = Callbacks()
        callback \
            .set_algorithm(ConvolutionalChannelsMovementAlgorithm(model=model).with_step(step).with_size(size)) \
            .set_validation_data(validation_data) \
            .set_size(size) \
            .set_step(step) \
            .set_validation_frequency(1)

        model.fit_generator(loader(), epochs=50, steps_per_epoch=25, shuffle=True, callbacks=[callback])
        model.save('conv_chan_movement_model_20000-7x7.h5')

# K: 10 LR: 0.001 SpE = 50
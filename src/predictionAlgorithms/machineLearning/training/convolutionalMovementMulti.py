from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD
import numpy as np
import os

from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalMovementMulti8Algorithm import \
    ConvolutionalMovementMulti8Algorithm
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalMovementMultiAlgorithm import \
    ConvolutionalMovementMultiAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.callbacks import Callbacks
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalMovementMulti:

    @staticmethod
    def train(data, size, channels, validation_data, output_size=4):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)

        # model = Sequential()
        # model.add(Conv2D(filters=8, kernel_size=(3, 3),
        #                    input_shape=(None, 40, 40, 1),
        #                    padding='same', return_sequences=True))

        # model = Sequential([
        #     Conv2D(filters=8,
        #            kernel_size=(4, 4),
        #            activation='relu',
        #            data_format='channels_first',
        #            input_shape=(channels, size, size)),
        #     Conv2D(filters=output_size,
        #            kernel_size=(6, 6),
        #            activation='relu',
        #            data_format='channels_first')
        # ])

        callback = Callbacks()
        callback\
            .set_algorithm(ConvolutionalMovementMulti8Algorithm())\
            .set_validation_data(validation_data)\
            .set_validation_frequency(1)

        model.compile(
            optimizer=SGD(lr=0.04),
            loss='mse'
        )

        model.fit(x, y, epochs=6, callbacks=[callback])
        model.save('cnn_movement_multi_8_model.h5')

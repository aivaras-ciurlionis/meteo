import tensorflow
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, ConvLSTM2D
from keras.optimizers import SGD
import numpy as np
import os
from keras import backend as K

from src.predictionAlgorithms.machineLearning.algorithms.ConvLSTM import ConvLstm
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.callbacks import Callbacks
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalLstmTrain:

    @staticmethod
    def train(size, channels, validation_data, loader, val):
        model = Sequential()
        model.add(
            ConvLSTM2D(
                   filters=1,
                   padding='same',
                   kernel_size=(6, 6),
                   activation='relu',
                   input_shape=(channels, 1, size, size),
                   data_format='channels_first',
                   return_sequences=False
            )
        )
        model.add(
            Conv2D(
                filters=1,
                kernel_size=(8, 8),
                activation='relu',
                padding='same',
                data_format='channels_first'
            )
        )
        model.compile(
            optimizer=SGD(lr=0.01, decay=0.01/50),
            loss='mse'
        )
        callback = Callbacks()
        callback \
            .set_algorithm(ConvLstm(model=model).with_size(size)) \
            .set_validation_data(validation_data) \
            .set_size(size) \
            .set_validation_frequency(1) \
            .set_base(6)

        model.fit_generator(loader(), epochs=50, steps_per_epoch=20, shuffle=True, callbacks=[callback],
                            validation_data=val)
        model.save('conv_lstm.h5')

# K: 12x12 -> lr: 0.01 -> E = 50; SpE = 10
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
import keras

class ConvolutionalWithChannelsMovement:

    @staticmethod
    def train(size, channels, validation_data, loader, step, val):
        model = Sequential()
        model.add(
            Conv2D(
                   filters=4,
                   kernel_size=(8, 8),
                   padding='same',
                   activation='relu',
                   input_shape=(channels, size, size),
                   data_format='channels_first')
        )
        model.add(
            Conv2D(
                filters=2,
                kernel_size=(4, 4),
                padding='same',
                activation='relu',
                input_shape=(channels, size, size),
                data_format='channels_first')
        )
        model.add(
            Conv2D(
                filters=1,
                kernel_size=(4, 4),
                padding='same',
                activation='relu',
                data_format='channels_first')
        )
        model.compile(
            optimizer=SGD(lr=0.01),
            loss='mse'
        )

        callback = Callbacks()
        callback \
            .set_algorithm(ConvolutionalChannelsMovementAlgorithm(model=model).with_step(step).with_size(size)) \
            .set_validation_data(validation_data) \
            .set_size(size) \
            .set_step(step) \
            .set_validation_frequency(1)

        keras.callbacks.ModelCheckpoint('conv_chan_movement_model_20000-128-8x8-6x6-4x4.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=1)

        model.fit_generator(loader(), epochs=20, steps_per_epoch=224, shuffle=True, callbacks=[callback], validation_data=val)
        model.save('conv_chan_movement_model_20000-128-4x4-6x6.h5')

# K: 10 LR: 0.001 SpE = 50

# 128-8x8(6)-3x3(1)-6x6(1)
# from src.predictionAlgorithms.machineLearning.helpers.channelsInputLoader import ChannelsInputLoader
# from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
# from src.predictionAlgorithms.machineLearning.training.convolutionalWithChannels import ConvolutionalWithChannels
# from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
#
#
# channelsInputLoader = ChannelsInputLoader()
#
# channelsInputLoader.select_folder('../channels_input').set_sample_size(5)
# data = channelsInputLoader.load_data()
# ConvolutionalWithChannels.train(data, 64, 4)\
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import SGD
import tensorflow
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import SGD
import numpy as np
import os
from keras import backend as K

from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.callbacks import Callbacks
from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor

from src.predictionAlgorithms.machineLearning.training.convolutionalLstm import ConvolutionalLstmTrain
from src.predictionAlgorithms.machineLearning.training.convolutionalWithChannelsMovement import \
    ConvolutionalWithChannelsMovement
from src.utilities.fileProcessing.BinaryProcessor import BinaryProcessor
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor


def imageLoader():
    folder = '../binary_images/2019-08-20-128'
    shapeX = 'x-2019-08-20-128(100, 4, 128, 128)-'
    shapeY = 'y-2019-08-20-128(100, 1, 128, 128)-'
    while True:
        file_i = 0
        while file_i < 224:
            x = np.load(folder + '/' + shapeX + str(file_i) + '.npy')
            y = np.load(folder + '/' + shapeY + str(file_i) + '.npy')
            yield (x, y)
            file_i += 1


image_preprocessor = ImagePreprocessor()
evaluator = MultiAlgorithmAccuracyEvaluator()
evaluation_processor = EvaluationProcessor()
processor = SequenceProcessor()

#
validation_sequences = image_preprocessor\
    .set_images_folder('../pics')\
    .set_resized_image_dimension(128)\
    .set_max_images_per_sequence(50)\
    .set_date_range('2017-10-28 09:15', '2017-10-28 23:00')\
    .load_and_process_images()

# training_sequences = image_preprocessor\
#     .set_images_folder('../pics-full/MeteoData/Data')\
#     .set_crop_amount(0)\
#     .set_resized_image_dimension(128)\
#     .set_max_images_per_sequence(1000)\
#     .set_date_range('2017-11-01 05:00', '2017-12-31 14:00')\
#     .load_and_process_images()
#
# data = processor.set_sequences(training_sequences).merge_sequences_to_channels(4, 1, 128, '128-4-1-2017')

# data = processor.set_sequences(training_sequences)\
#     .merge_sequences_to_channels(4, 4, 128, 'sq4-2018', True)

x = np.load('../binary_images/128-4-1-2017/x-128-4-1-2017(1000, 4, 128, 128)-0.npy')
y = np.load('../binary_images/128-4-1-2017/y-128-4-1-2017(1000, 1, 128, 128)-0.npy')
#
#
# ConvolutionalLstmTrain.train(128, 6, validation_sequences, imageLoader, (x,y))

ConvolutionalWithChannelsMovement.train(128, 4, validation_sequences, imageLoader, 1, (x, y))

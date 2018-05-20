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


from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
from src.predictionAlgorithms.machineLearning.training.convolutionalMovementMulti import ConvolutionalMovementMulti
from src.predictionAlgorithms.machineLearning.training.convolutionalWithChannelsMovement import \
    ConvolutionalWithChannelsMovement
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor

image_preprocessor = ImagePreprocessor()
evaluator = MultiAlgorithmAccuracyEvaluator()
evaluation_processor = EvaluationProcessor()
processor = SequenceProcessor()

validation_sequences = image_preprocessor\
    .set_images_folder('../pics')\
    .set_resized_image_dimension(64)\
    .set_max_images_per_sequence(500)\
    .set_date_range('2017-10-28 03:30', '2017-10-30 23:00')\
    .load_and_process_images()


training_sequences = image_preprocessor\
    .set_images_folder('../pics2')\
    .set_resized_image_dimension(64)\
    .set_max_images_per_sequence(200)\
    .set_date_range('2017-11-01 05:00', '2018-04-25 14:00')\
    .load_and_process_images()

data = processor.set_sequences(training_sequences).merge_sequences_to_channels(4, 1)
ConvolutionalWithChannelsMovement.train(data, 64, 4, validation_sequences)

#data = processor.set_sequences(training_sequences).merge_sequences_to_channels(8, 8)
#ConvolutionalMovementMulti.train(data, 64, 8, validation_sequences, 8)
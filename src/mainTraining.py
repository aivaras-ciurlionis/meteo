from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
from src.predictionAlgorithms.machineLearning.training.convolutionalWithChannels import ConvolutionalWithChannels
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor


image_preprocessor = ImagePreprocessor()
sequences = image_preprocessor \
    .set_images_folder('../pics') \
    .set_resized_image_dimension(64) \
    .set_max_images_per_sequence(500) \
    .set_date_range('2017-10-23 01:15', '2017-10-30 23:00') \
    .load_and_process_images()

first_sequence = sequences[0]

sequence_preprocessor = SequenceProcessor()
prepared_data = sequence_preprocessor \
    .set_sequence(first_sequence) \
    .merge_to_channels(channel_count=3)

print(len(prepared_data[0]))

ConvolutionalWithChannels.train(prepared_data, 64, 3)

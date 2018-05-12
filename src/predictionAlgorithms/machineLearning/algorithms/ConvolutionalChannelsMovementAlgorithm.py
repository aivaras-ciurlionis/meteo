from PIL import Image
from keras.models import load_model
import numpy as np

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.channelsInputLoader import ChannelsInputLoader
from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
from src.predictionAlgorithms.sequenceCorelation.multiImageSequenceTransformation import \
    MultiImageSequenceTransformation
from src.predictionAlgorithms.transformations import Transformations
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.errorFunctions import trueSkillStatistic
from PIL import ImageChops as iC

def offset(image, x, y):
    s = image.size[0]
    x = int(round(x))
    y = int(round(y))
    i = iC.offset(image, x, y)
    if x > 0:
        i.paste(0, (0, 0, x, s))
    else:
        i.paste(0, (s+x, 0, s, s))

    if y > 0:
        i.paste(0, (0, 0, s, y))
    else:
        i.paste(0, (0, s-y, s, s))
    return i


class ConvolutionalChannelsMovementAlgorithm(BaseAlgorithm):
    model = None
    name = 'Conv channels movement'

    def __init__(self):
        super().__init__()
        self.model = load_model('conv_chan_movement_model.h5')

    def predict(self, source_images, count):
        processor = SequenceProcessor()
        source_images = source_images[-4:]
        converted_images = PixelsRainStrengthConverter.convert_loaded(source_images)
        merged_images = ChannelsInputLoader.merge_images(converted_images)
        merged_images = np.asarray([merged_images])

        results = []
        for i in range(0, count):
            result_image = self.model.predict(merged_images)[0][0]

            next_image = Image.new('L', (56, 56))
            next_image.putdata(result_image.flatten())
            next_image = next_image.resize((64, 64))
            next_data = np.asarray(next_image.getdata())

            next_data = next_data.reshape((64, 64))

            imges = np.asarray([merged_images[0][-3], merged_images[0][-2], merged_images[0][-1], next_data])

            merged_images = ChannelsInputLoader.merge_images(imges)
            merged_images = np.asarray([merged_images])

            r = np.array(list(map(lambda y: int(y) * 16, next_image.getdata())))
            img = Image.new('L', (64, 64))
            img.putdata(r)
            results.append(img)

        return results

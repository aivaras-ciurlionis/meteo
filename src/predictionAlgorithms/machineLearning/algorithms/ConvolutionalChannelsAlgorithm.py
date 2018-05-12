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


class ConvolutionalChannelsAlgorithm(BaseAlgorithm):
    model = None
    name = 'Conv channels'

    def __init__(self):
        super().__init__()
        self.model = load_model('conv_chan_model.h5')

    def predict(self, source_images, count):
        processor = SequenceProcessor()
        source_images = source_images[-4:]
        print(len(source_images))

        algorithm = MultiImageSequenceTransformation(Transformations.xy_transformation(),
                                                     trueSkillStatistic.TrueSkillStatistic(),4)

        movement_vector = algorithm.find_best_movement_vector_multi(source_images[-4].copy(), source_images[-4 + 1:])
        print(movement_vector)
        x_images = []
        for j, img in enumerate(source_images):
            next_img = offset(img, -movement_vector[0] * j, -movement_vector[1] * j)
            converted_image = PixelsRainStrengthConverter.convert_gray_strength_to_source(next_img)
            x_images.append(converted_image)
        converted_images = PixelsRainStrengthConverter.convert_images(x_images, True)
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

            result_image = result_image.flatten()
            r = np.array(list(map(lambda y: int(y) * 16, result_image)))
            img = Image.new('L', (56, 56))
            img.putdata(r)
            image = Image.new('L', (64, 64))
            offset_x = int(movement_vector[0] * (i+4))
            offset_y = int(movement_vector[1] * (i+4))
            image.paste(img, (offset_x, offset_y))
            results.append(image)

        return results

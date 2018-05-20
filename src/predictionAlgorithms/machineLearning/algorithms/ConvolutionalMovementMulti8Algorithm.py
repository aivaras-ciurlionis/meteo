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


class ConvolutionalMovementMultiAlgorithm(BaseAlgorithm):
    model = None
    name = 'Conv channels movement multi'

    def __init__(self):
        super().__init__()
        self.model = load_model('epoch 4.h5')

    def reload(self, model_file='conv_chan_movement_model.h5'):
        self.model = load_model(model_file)

    def predict(self, source_images, count):
        source_images = source_images[-4:]
        converted_images = PixelsRainStrengthConverter.convert_loaded(source_images)
        merged_images = ChannelsInputLoader.merge_images(converted_images)
        merged_images = np.asarray([merged_images])
        results = []
        result_images = self.model.predict(merged_images)[0]

        for image in result_images:
            r = np.array(list(map(lambda y: int(y) * 16,  image.flatten())))
            print(r.shape)
            next_image = Image.new('L', (56, 56))
            next_image.putdata(r)
            img = next_image.resize((64, 64))
            results.append(img)

        return results

from PIL import Image
from keras.models import load_model
import numpy as np

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalChannelsAlgorithm(BaseAlgorithm):

    def __init__(self):
        super().__init__()

    def predict(self, source_images, count):
        processor = SequenceProcessor()
        images = processor.set_sequence(source_images[:5]).merge_to_channels(3)
        model = load_model('conv_chan_model.h5')
        x = np.asarray(images[0])
        print(len(x))
        r = model.predict(x)
        print(r)
        r = np.array(list(map(lambda y: y * 255, r)))

        r.flatten()

        image = Image.new('RGBA', (64, 64))
        image.putdata(r)
        return [PixelsRainStrengthConverter.convert_gray_strength_to_source(image)]
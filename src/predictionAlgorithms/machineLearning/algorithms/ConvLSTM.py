from PIL import Image
from keras.models import load_model
import numpy as np
from keras import backend as K

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.predictionAlgorithms.machineLearning.helpers.channelsInputLoader import ChannelsInputLoader
from src.predictionAlgorithms.machineLearning.helpers.sequenceProcessor import SequenceProcessor
from src.predictionAlgorithms.sequenceCorelation.multiImageSequenceTransformation import \
    MultiImageSequenceTransformation
from src.predictionAlgorithms.transformations import Transformations
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.errorFunctions import trueSkillStatistic
from PIL import ImageChops as iC
import math


class ConvLstm(BaseAlgorithm):
    model = None
    name = 'convLSTM'

    def __init__(self, file='/app/src/savedModels/convLSTM_96_p2_CNN_64_u2_3_3'):
        self.model = self.load_ml_model(file)

    def reload(self, file='/app/src/savedModels/convLSTM_96_p2_CNN_64_u2_3_3'):
        self.model = self.load_ml_model(file)

    @staticmethod
    def remove_rain_enhancement(p):
        p = round(p)
        return p * 16

    def predict(self, source_images, count):
        print('Predict ', self.name)
        converted_images = PixelsRainStrengthConverter.convert_loaded(source_images[-4:])
        window = np.array(converted_images)
        results = []
        for i in range(count):
            print('generating image ' + str(i))
            temp = np.copy(window[:4])[np.newaxis, ...]
            print(temp.shape)
            temp_expanded = self.get_model_input(temp, 0, 0, True)
            forecast = self.model.predict(temp_expanded)
            window[:-1] = window[1:]
            window[-1] = np.copy(forecast)
            r = np.array(list(map(ConvLstm.remove_rain_enhancement, forecast.flatten())))
            img = Image.new('L', (self.size, self.size))
            img.putdata(r)
            resized = img.resize((128, 128), Image.BILINEAR)
            results.append(resized)
        return results

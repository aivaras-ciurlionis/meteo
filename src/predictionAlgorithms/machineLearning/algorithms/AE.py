from PIL import Image

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
import numpy as np


class AE(BaseAlgorithm):
    model = None
    name = 'AE'

    def __init__(self, file='/app/src/savedModels/ae_3rand_1elev_32_p2_64_p2_128_u2_64_u2_16_3_3'):
        self.model = self.load_ml_model(file)

    def reload(self, file='/app/src/savedModels/ae_3rand_1elev_32_p2_64_p2_128_u2_64_u2_16_3_3'):
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
            temp_expanded = self.get_model_input(temp, 1, 3)
            forecast = self.model.predict(temp_expanded)
            window[:-1] = window[1:]
            window[-1] = np.copy(forecast)
            r = np.array(list(map(AE.remove_rain_enhancement, forecast.flatten())))
            img = Image.new('L', (self.size, self.size))
            img.putdata(r)
            resized = img.resize((128, 128), Image.BILINEAR)
            results.append(resized)
        return results

from PIL import Image

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
import numpy as np
from os import listdir, curdir

class CNN4L(BaseAlgorithm):
    model = None
    name = 'CNN 4 Layers'

    def __init__(self,file='/app/src/savedModels/3l_3rand_1elev_96_64_3_3'):
        self.model = self.load_ml_model(file)
        # self.model = None

    def reload(self, file='/app/src/savedModels/3l_3rand_1elev_96_64_3_3'):
        self.model = self.load_ml_model(file)

    def predict(self, source_images, count):
        print('Predict ', self.name)
        converted_images = PixelsRainStrengthConverter.convert_loaded(source_images[-4:])
        window = np.array(converted_images)
        print(np.max(window[0]))
        print(np.mean(window[0]))
        results = []
        for i in range(count):
            print('generating image ' + str(i))
            temp = np.copy(window[:4])[np.newaxis, ...]
            print('w', np.max(window))
            print('w', np.mean(window))
            print(temp.shape)
            temp_expanded = self.get_model_input(temp, 1, 3)
            forecast = self.model.predict(temp_expanded)
            window[:-1] = window[1:]
            window[-1] = np.copy(forecast)
            img = Image.new('L', (self.size, self.size))
            img.putdata(forecast.flatten())
            resized = img.resize((128, 128), Image.BILINEAR)
            results.append(resized)
            print('mx', np.max(forecast))
        return results

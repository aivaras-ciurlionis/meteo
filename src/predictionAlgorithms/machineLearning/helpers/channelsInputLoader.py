import os
import numpy as np

from os import path

from PIL import Image

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer


class ChannelsInputLoader:
    srcFolder = ''
    sample_size = 5

    def select_folder(self, folder_name):
        self.srcFolder = folder_name
        return self

    def set_sample_size(self, size):
        self.sample_size = size

    def load_data(self):
        converter = PixelsRainStrengthConverter()
        results_x = []
        results_y = []
        files = os.listdir(self.srcFolder)
        files.sort()
        for i in range(0, len(files), self.sample_size):
            sample = []
            for j in range(0, self.sample_size):
                image_location = path.join(self.srcFolder, files[i+j])
                image = Image.open(image_location)
                sample.append(image)

            converted_images = converter.convert_images(sample, True)
            print('loaded: ', i / len(files))

            x_images = converted_images[0:self.sample_size-1]

            y_image_data = [np.asarray(converted_images[-1])[4:60, 4:60]]

            merged = self.merge_images(x_images)
            results_x.append(merged)
            results_y.append(y_image_data)

        return results_x, results_y

    @staticmethod
    def merge_images(images):
        merged = []
        for image in images:
            data = np.asarray(image)
            merged.append(data)
        return merged
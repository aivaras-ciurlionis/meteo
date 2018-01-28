import json

import arrow
import os

from src.prediction.singleAlgorithmPrediction import SingleAlgorithmPrediction
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class MultiAlgorithmPrediction:
    source_date = ''
    resize_size = 128
    images_folder = ''
    algorithms = []
    prediction_count = 8
    output_dir = ''

    DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'
    prediction_results = None

    def set_output_dir(self, dir):
        self.output_dir = dir
        return self

    def set_source_date(self, date):
        self.source_date = date
        return self

    def set_images_folder(self, src):
        self.images_folder = src
        return self

    def set_resize_size(self, size):
        self.resize_size = size
        return self

    def set_algorithms(self, algorithms):
        self.algorithms = algorithms
        return self

    def set_predicted_images(self, count):
        self.prediction_count = count
        return self

    def save_images(self, images, prefix):
        saved_names = []
        for index, image in enumerate(images):
            src = os.path.join(self.output_dir, prefix + str(index) + '.png')
            saved_names.append(src)
            converted_image = PixelsRainStrengthConverter.convert_gray_strength_to_source(image)
            converted_image.save(src)
        return saved_names

    def load_source_images(self):
        preprocessor = ImagePreprocessor()
        end_date = arrow.get(self.source_date)
        start_date = end_date.shift(hours=-1).format(self.DATE_FORMAT)
        images = preprocessor.set_resized_image_dimension(self.resize_size) \
            .set_date_range(start_date, end_date) \
            .set_images_folder(self.images_folder) \
            .load_and_process_images()[0]
        return images

    def load_actual_images(self):
        preprocessor = ImagePreprocessor()
        start_date = arrow.get(self.source_date).shift(minutes=15)
        end_date = arrow.get(self.source_date).shift(minutes=15*self.prediction_count)
        images = preprocessor.set_resized_image_dimension(self.resize_size) \
            .set_date_range(start_date, end_date) \
            .set_images_folder(self.images_folder) \
            .load_and_process_images()[0]
        return images

    def predict(self):
        results = []
        source_images = self.load_source_images()
        actual_images = self.load_actual_images()

        file_names = self.save_images(actual_images, 'actual')
        results.append(dict(
            files=file_names,
            accuracy=None,
            name='Actual'
        ))

        for index, algorithm in enumerate(self.algorithms):
            prediction = SingleAlgorithmPrediction()
            gen_images, accuracy = prediction.predict(algorithm, source_images, actual_images, self.prediction_count)
            saved_names = self.save_images(gen_images, algorithm.name)
            results.append(dict(
                files=saved_names,
                accuracy=accuracy,
                name=algorithm.name
            ))

        self.prediction_results = results
        return self

    def dump_to_json(self, output):
        if self.prediction_results is not None:
            json_string = json.dumps(self.prediction_results)
            src = os.path.join(output, 'result.json')
            with open(src, 'w') as text_file:
                text_file.write(json_string)

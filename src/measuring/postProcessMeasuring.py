import json

import arrow

from src.measuring.partAccuracyEvaluator import PartAccuracyEvaluator
from src.utilities.fileProcessing.ImageLoader import ImageLoader
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer


class PostProcessMeasuring:
    files = []
    DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'
    error_function = None

    def set_files(self, files):
        self.files = files
        return self

    def set_error_function(self, error_function):
        self.error_function = error_function
        return self

    def evaluate(self):
        image_loader = ImageLoader()
        converter = PixelsRainStrengthConverter()
        image_resizer = ImageResizer()
        gen_images = image_loader \
            .set_image_folder('../output') \
            .set_sequence(self.files) \
            .set_max_images(500) \
            .load_sequence_images()
        generated_images = converter.convert_images(gen_images)
        accuracies = []
        for i, img in enumerate(generated_images):
            parts = self.files[i].split('_')
            actual_date = parts[1]
            prediction_minutes = int(parts[2].split('m')[0])
            actual_time = arrow.get(actual_date).shift(minutes=prediction_minutes)
            formatted = actual_time.format(self.DATE_FORMAT)
            actual = image_loader\
                .set_image_folder('../meteo-out/actual') \
                .set_sequence([formatted+'.png']) \
                .set_max_images(1) \
                .load_sequence_images()

            actual = converter.convert_images(actual)

            image_resizer \
                .set_images(actual) \
                .resize_images(img.size)

            accuracy = PartAccuracyEvaluator.evaluate_part_accuracy(
                actual,
                [img],
                self.error_function
            )[0]
            accuracies.append(dict(
                file=self.files[i],
                accuracy=accuracy)
            )
        return json.dumps(accuracies)

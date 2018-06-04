import os


from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ImagesPrediction:
    resize_size = 128
    source_folder = ''
    algorithms = []
    algorthmNames = []
    prediction_count = 8
    output_dir = ''
    DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'
    source_date = ''
    prediction_results = None

    def set_source_date(self, date):
        self.source_date = date
        return self

    def set_output_dir(self, dir):
        self.output_dir = dir
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

    def set_algorithm_names(self, names):
        self.algorthmNames = names
        return self

    def load_source_images(self):
        preprocessor = ImagePreprocessor()
        images = preprocessor.set_resized_image_dimension(self.resize_size) \
            .set_images_folder(self.images_folder) \
            .load_and_process_images()[0]
        return images

    def save_images(self, images, prefix):
        saved_names = []
        for index, image in enumerate(images):
            filename = prefix + '_' + self.source_date + '_' + str((index + 1) * 15) + 'm_' + '.png'
            src = os.path.join(self.output_dir, filename)
            saved_names.append(filename)
            converted_image = PixelsRainStrengthConverter.convert_gray_strength_to_source(image)
            converted_image.save(src)
        return saved_names

    def predict(self):
        source_images = self.load_source_images()
        results = []
        for index, algorithm in enumerate(self.algorithms):
            generated_images = algorithm.predict(source_images, self.prediction_count)
            saved_names = self.save_images(generated_images, self.algorthmNames[index])
            results.append(dict(
                files=saved_names,
                name=self.algorthmNames[index]
            ))
        return results

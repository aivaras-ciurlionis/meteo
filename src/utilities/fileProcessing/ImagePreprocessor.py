from src.utilities.fileProcessing.ImageLoader import ImageLoader
from src.utilities.fileProcessing.ImageSequencesLoader import ImageSequencesLoader
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer


class ImagePreprocessor:
    imagesFolder = ''
    resizedImageDimension = 128
    maxImagesPerSequence = 10000

    def set_images_folder(self, folder):
        self.imagesFolder = folder
        return self

    def set_max_images_per_sequence(self, max_images):
        self.maxImagesPerSequence = max_images
        return self

    def set_resized_image_dimension(self, value):
        self.resizedImageDimension = value
        return self

    def load_and_process_images(self):
        sequences_loader = ImageSequencesLoader()
        image_loader = ImageLoader()
        image_resizer = ImageResizer()
        converter = PixelsRainStrengthConverter()

        prepared_sequences = []
        image_sequences = sequences_loader.select_folder(self.imagesFolder).load_sequences()

        for sequence in image_sequences:
            sequence_images = image_loader\
                .set_image_folder(self.imagesFolder)\
                .set_sequence(sequence)\
                .set_max_images(self.maxImagesPerSequence)\
                .load_sequence_images()

            image_resizer\
                .set_images(sequence_images)\
                .resize_images((self.resizedImageDimension, self.resizedImageDimension))

            converted_images = converter.convert_images(sequence_images)
            prepared_sequences.append(converted_images)

        return prepared_sequences

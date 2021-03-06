from src.utilities.fileProcessing.ImageLoader import ImageLoader
from src.utilities.fileProcessing.ImageSequencesLoader import ImageSequencesLoader
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer


class ImagePreprocessor:
    imagesFolder = ''
    resizedImageDimension = 128
    cropAmount = 0
    maxImagesPerSequence = 10000
    startDate = None
    endDate = None

    def set_crop_amount(self, amount):
        self.cropAmount = amount
        return self

    def set_images_folder(self, folder):
        self.imagesFolder = folder
        return self

    def set_date_range(self, start_date, end_date):
        self.startDate = start_date
        self.endDate = end_date
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
        image_sequences = sequences_loader\
            .select_folder(self.imagesFolder)\
            .set_date_range(self.startDate, self.endDate)\
            .load_sequences()

        for sequence in image_sequences:
            sequence_images = image_loader\
                .set_image_folder(self.imagesFolder)\
                .set_sequence(sequence)\
                .set_max_images(self.maxImagesPerSequence)\
                .load_sequence_images()
            print('resizing')
            image_resizer\
                .set_images(sequence_images)\
                .set_crop_amount(self.cropAmount)\
                .resize_images((self.resizedImageDimension, self.resizedImageDimension))
            print('resizing done')
            print('converting')
            converted_images = converter.convert_images(sequence_images)
            prepared_sequences.append(converted_images)

        return prepared_sequences

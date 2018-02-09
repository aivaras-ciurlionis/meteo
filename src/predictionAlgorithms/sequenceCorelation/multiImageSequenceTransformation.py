import numpy

from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation
from src.utilities.imageAnalysis.imagesMeanSquareError import ImagesMeanSquareError
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class MultiImageSequenceTransformation(BaseTransformation):
    transformations = []
    name = 'Multi image sequence'
    source_count = 4

    def __init__(self, transformation_algorithm, source_count=4):
        self.name += ' ' + transformation_algorithm[0]
        self.transformations = transformation_algorithm[1]
        self.source_count = source_count
        super().__init__(transformation_algorithm)

    def predict(self, source_images, count):
        best_vector = self.find_best_movement_vector_multi(source_images[-self.source_count].copy(), source_images[-self.source_count+1:])
        print(best_vector)
        return self.generate_images(source_images, best_vector, count)

    def find_best_movement_vector_multi(self, start_image, evaluation_images):
        base_vector = numpy.zeros(len(self.transformations))
        current_vector = numpy.zeros(len(self.transformations))
        return self\
            .find_vector_recursive(start_image, evaluation_images, 0, 100, base_vector, current_vector)

    def find_vector_recursive(self, start_image, evaluation_images, index, best_error, best_vector, current_vector):
        if index >= len(self.transformations):
            return best_vector

        value = self.transformations[index][1][0]
        end = self.transformations[index][1][1]
        step = 1
        if len(self.transformations[index][1]) > 2:
            step = self.transformations[index][1][2]

        while value < end:
            best_vector = self\
                .find_vector_recursive(start_image, evaluation_images, index + 1, best_error, best_vector, current_vector)
            current_vector[index] = value
            error = self.find_current_error(start_image, evaluation_images, current_vector)
            value += step
            if error < best_error:
                best_vector[index] = value
                best_error = error
        return best_vector

    def find_current_error(self, start_image, evaluation_images, current_vector):
        generated = self.generate_images([start_image], current_vector, self.source_count-1)
        error = 0
        for i, image in enumerate(generated):
            image1 = PixelsRainStrengthConverter.normalise_image(image)
            image2 = PixelsRainStrengthConverter.normalise_image(evaluation_images[i])
            step_error = ImagesMeanSquareError.get_mean_square_error(image1, image2)
            error += step_error
        return error

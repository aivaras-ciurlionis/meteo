import numpy

from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class MultiImageSequenceTransformation(BaseTransformation):
    transformations = []
    baseName = 'Multi image sequence'
    name = ''
    source_count = 4
    errorFunction = None

    def __init__(self, transformation_algorithm, error_function, source_count=4):
        self.name = self.baseName + ' ' + transformation_algorithm[0]
        self.transformations = transformation_algorithm[1]
        self.source_count = source_count
        self.errorFunction = error_function
        super().__init__(transformation_algorithm, error_function)

    def predict(self, source_images, count):
        best_vector = self.find_best_movement_vector_multi(source_images[-self.source_count].copy(), source_images[-self.source_count+1:])
        print(best_vector)
        return self.generate_images(source_images, best_vector, count)

    def find_best_movement_vector_multi(self, start_image, evaluation_images):
        base_vector = numpy.zeros(len(self.transformations))
        current_vector = numpy.zeros(len(self.transformations))
        print('-------')
        return self\
            .find_vector_recursive(start_image, evaluation_images, 0, -100, base_vector, current_vector)[0]

    def find_vector_recursive(self, start_image, evaluation_images, index, best_error, best_vector, current_vector):
        if index >= len(self.transformations):
            return best_vector, best_error
        value = self.transformations[index][1][0]
        end = self.transformations[index][1][1]
        step = 1
        if len(self.transformations[index][1]) > 2:
            step = self.transformations[index][1][2]
        current_vector[index] = value
        while value < end:
            result = self\
                .find_vector_recursive(start_image, evaluation_images, index + 1, best_error, best_vector, current_vector)
            best_vector = result[0]
            best_error = result[1]
            value += step
            current_vector[index] = value
            error = self.find_current_error(start_image, evaluation_images, current_vector)
            if error > best_error:
                best_vector = list(current_vector)
                best_error = error
        return best_vector, best_error

    def find_current_error(self, start_image, evaluation_images, current_vector):
        generated = self.generate_images([start_image], current_vector, self.source_count-1)
        error = 0
        for i, image in enumerate(generated):
            image1 = PixelsRainStrengthConverter.normalise_image(image)
            image2 = PixelsRainStrengthConverter.normalise_image(evaluation_images[i])
            step_error = self.errorFunction.get_error(image1, image2)
            error += step_error
        return error / len(evaluation_images)

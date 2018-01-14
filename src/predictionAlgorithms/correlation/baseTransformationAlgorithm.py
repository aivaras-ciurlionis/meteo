import numpy

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.utilities.imageAnalysis.imagesMeanSquareError import ImagesMeanSquareError
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class BaseTransformationAlgorithm(BaseAlgorithm):
    transformations = []

    def predict(self, source_images, count):
        best_vector = self.find_best_movement_vector(source_images)
        print(best_vector)
        return self.generate_images(source_images, best_vector, count)

    def find_best_movement_vector(self, source_images):
        base_vector = numpy.zeros(len(self.transformations))
        return self.find_vector_recursive(source_images, 0, 100, base_vector)

    def find_vector_recursive(self, images, index, best_error, current_vector):
        if index >= len(self.transformations):
            return current_vector

        value = self.transformations[index][1][0]
        end = self.transformations[index][1][1]
        step = 1
        if len(self.transformations[index][1]) > 2:
            step = self.transformations[index][1][2]
        algorithm = self.transformations[index][0]
        working_image = images[-2].copy()
        evaluation_image = images[-1]
        working_image = algorithm(working_image, value)
        while value < end:
            current_vector = self.find_vector_recursive(images, index + 1, best_error, current_vector)
            working_image = algorithm(working_image, step)
            value += step
            image1 = PixelsRainStrengthConverter.normalise_image(working_image)
            image2 = PixelsRainStrengthConverter.normalise_image(evaluation_image)
            error = ImagesMeanSquareError.get_mean_square_error(image1, image2)
            if error < best_error:
                current_vector[index] = value
                best_error = error
        return current_vector

    def generate_images(self, images, best_movement_vector, count):
        generated_images = []
        working_image = images[-1].copy()
        for index in range(count):
            for i, transformation in enumerate(self.transformations):
                algorithm = transformation[0]
                working_image = algorithm(working_image, int(best_movement_vector[i]))
            generated_images.append(working_image.copy())
        return generated_images

import numpy

from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm
from src.utilities.imageAnalysis.imagesMeanSquareError import ImagesMeanSquareError
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class BaseTransformation(BaseAlgorithm):
    name = 'Two frame transform'
    transformations = []

    def __init__(self, transformation_algorithm):
        self.name += ' ' + transformation_algorithm[0]
        self.transformations = transformation_algorithm[1]
        super().__init__()

    def predict(self, source_images, count):
        best_vector = self.find_best_movement_vector(source_images, source_images[-2].copy(), source_images[-1])
        print(best_vector)
        return self.generate_images(source_images, best_vector, count)

    def find_best_movement_vector(self, source_images, working_image, evaluation_image):
        base_vector = numpy.zeros(len(self.transformations))
        return self\
            .find_vector_recursive(source_images, working_image, evaluation_image, 0, 100, base_vector)

    def find_vector_recursive(self, images, working_image, evaluation_image, index, best_error, best_vector):
        if index >= len(self.transformations):
            return best_vector

        value = self.transformations[index][1][0]
        end = self.transformations[index][1][1]
        step = 1
        if len(self.transformations[index][1]) > 2:
            step = self.transformations[index][1][2]
        algorithm = self.transformations[index][0]
        working_image = algorithm(working_image, value)
        while value < end:
            best_vector = self\
                .find_vector_recursive(images, working_image.copy(), evaluation_image, index + 1, best_error, best_vector)
            working_image = algorithm(working_image, step)
            image1 = PixelsRainStrengthConverter.normalise_image(working_image)
            image2 = PixelsRainStrengthConverter.normalise_image(evaluation_image)
            error = ImagesMeanSquareError.get_mean_square_error(image1, image2)
            value += step
            if error < best_error:
                best_vector[index] = value
                best_error = error
        return best_vector

    def generate_images(self, images, best_movement_vector, count):
        generated_images = []
        working_image = images[-1].copy()
        for index in range(count):
            for i, transformation in enumerate(self.transformations):
                algorithm = transformation[0]
                working_image = algorithm(working_image, int(best_movement_vector[i]))
            generated_images.append(working_image.copy())
        return generated_images

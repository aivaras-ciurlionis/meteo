from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class PartAccuracyEvaluator:
    @staticmethod
    def evaluate_part_accuracy_in_point(actual_images, generated_images, point):
        accuracies = []
        for index, image in enumerate(actual_images):
            strength_at_point_generated = PixelsRainStrengthConverter \
                .get_normalised_accuracy_in(generated_images[index], point)
            strength_at_point_actual = PixelsRainStrengthConverter \
                .get_normalised_accuracy_in(actual_images[index], point)
            error = abs(strength_at_point_generated - strength_at_point_actual)
            accuracies.append(error)
        return accuracies

    @staticmethod
    def evaluate_part_accuracy(actual_images, generated_images, error_function):
        accuracies = []
        for index, image in enumerate(actual_images):
            normalised_generated = PixelsRainStrengthConverter.normalise_image(generated_images[index])
            normalised_actual = PixelsRainStrengthConverter.normalise_image(image)
            error = error_function.get_error(normalised_generated, normalised_actual)
            accuracies.append(error)
        return accuracies

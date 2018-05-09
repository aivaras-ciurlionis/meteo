from src.measuring.partAccuracyEvaluator import PartAccuracyEvaluator


class SingleAlgorithmPrediction:

    @staticmethod
    def predict(algorithm, source_images, actual_images, count, error_function):
        generated_images = algorithm.predict(source_images, count)
        image_accuracies = PartAccuracyEvaluator.evaluate_part_accuracy(actual_images,
                                                                        generated_images,
                                                                        error_function)
        return generated_images, image_accuracies

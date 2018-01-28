from src.measuring.partAccuracyEvaluator import PartAccuracyEvaluator


class SingleAlgorithmPrediction:

    @staticmethod
    def predict(algorithm, source_images, actual_images, count):
        generated_images = algorithm.predict(source_images, count)
        image_accuracies = PartAccuracyEvaluator.evaluate_part_accuracy(actual_images, generated_images)
        return generated_images, image_accuracies

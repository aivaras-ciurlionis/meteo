from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.utilities.errorFunctions import imagesMeanSquareError


class MultiAlgorithmAccuracyEvaluator:
    imageSequences = []
    predictionSourceImagesCount = 1
    predictedImagesCount = 4
    predictionAlgorithms = []
    measuringType = 'image'
    measuringPoint = (0, 0)
    rangeStep = 1
    errorFunction = imagesMeanSquareError

    def set_image_sequences(self, sequences):
        self.imageSequences = sequences
        return self

    def set_error_function(self, error_function):
        self.errorFunction = error_function
        return self

    def set_predicted_images_count(self, count):
        self.predictedImagesCount = count
        return self

    def set_source_images_count(self, count):
        self.predictionSourceImagesCount = count
        return self

    def set_measuring_point(self, point):
        self.measuringPoint = point
        return self

    def set_range_step(self, step):
        self.rangeStep = step
        return self

    def set_measuring_type(self, measuring_type):
        self.measuringType = measuring_type
        return self

    def set_prediction_algorithms(self, algorithms):
        self.predictionAlgorithms = algorithms
        return self

    def evaluate(self):
        results = []
        evaluator = AccuracyEvaluator()
        for algorithm in self.predictionAlgorithms:
            result = evaluator \
                .set_image_sequences(self.imageSequences) \
                .set_predicted_images_count(self.predictedImagesCount) \
                .set_source_images_count(self.predictionSourceImagesCount) \
                .set_measuring_point(self.measuringPoint) \
                .set_range_step(self.rangeStep) \
                .set_error_function(self.errorFunction) \
                .set_measuring_type(self.measuringType) \
                .set_prediction_algorithm(algorithm) \
                .evaluate()
            results.append(result)
        return results

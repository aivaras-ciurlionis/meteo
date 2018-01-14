from src.utilities.imageAnalysis.imagesMeanSquareError import ImagesMeanSquareError
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class AccuracyEvaluator:
    imageSequences = []
    predictionSourceImagesCount = 1
    predictedImagesCount = 4
    predictionAlgorithm = None

    def set_image_sequences(self, sequences):
        self.imageSequences = sequences
        return self

    def set_predicted_images_count(self, count):
        self.predictedImagesCount = count
        return self

    def set_prediction_algorithm(self, algorithm):
        self.predictionAlgorithm = algorithm
        return self

    def set_prediction_source_images_count(self, count):
        self.predictionSourceImagesCount = count
        return self

    def evaluate_sequence(self, sequence):
        sequence_start = self.predictionSourceImagesCount - 1
        sequence_end = len(sequence) - self.predictedImagesCount
        sequence_accuracy = []
        for index in range(sequence_start, sequence_end):
            generated_images = self.predictionAlgorithm\
                .predict(sequence[index:index+self.predictionSourceImagesCount], self.predictedImagesCount)
            part_accuracy = self\
                .evaluate_part_accuracy(sequence[index+1:index+1+self.predictedImagesCount], generated_images)
            sequence_accuracy.append(part_accuracy)

        return sequence_accuracy

    @staticmethod
    def evaluate_part_accuracy(actual_images, generated_images):
        accuracies = []
        for index, image in enumerate(actual_images):
            normalised_generated = PixelsRainStrengthConverter.normalise_image(generated_images[index])
            normalised_actual = PixelsRainStrengthConverter.normalise_image(image)
            error = ImagesMeanSquareError.get_mean_square_error(normalised_generated, normalised_actual)
            accuracies.append(error)
        return accuracies

    def evaluate(self):
        overall_accuracy = []
        for sequence in self.imageSequences:
            sequence_accuracy = self.evaluate_sequence(sequence)
            overall_accuracy.append(sequence_accuracy)

        return overall_accuracy

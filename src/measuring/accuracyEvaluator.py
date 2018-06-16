from src.measuring.partAccuracyEvaluator import PartAccuracyEvaluator


class AccuracyEvaluator:
    imageSequences = []
    predictionSourceImagesCount = 1
    predictedImagesCount = 4
    predictionAlgorithm = None
    measuringType = 'image'
    measuringPoint = (0, 0)
    rangeStep = 1
    errorFunction = None

    def set_range_step(self, step):
        self.rangeStep = step
        return self

    def set_error_function(self, error_function):
        self.errorFunction = error_function
        return self

    def set_measuring_point(self, point):
        self.measuringPoint = point
        return self

    def set_image_sequences(self, sequences):
        self.imageSequences = sequences
        return self

    def set_measuring_type(self, measure_type):
        self.measuringType = measure_type
        return self

    def set_predicted_images_count(self, count):
        self.predictedImagesCount = count
        return self

    def set_source_images_count(self, count):
        self.predictionSourceImagesCount = count
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
        for index in range(sequence_start, sequence_end, self.rangeStep):
            generated_images = self.predictionAlgorithm\
                .predict(sequence[index+1-self.predictionSourceImagesCount:index+1], self.predictedImagesCount)
            if len(generated_images) < self.predictedImagesCount:
                self.predictionSourceImagesCount = len(generated_images)
            print(index / (sequence_end-sequence_start) * 100)
            actual_img = sequence[index + 1:index + 1 + self.predictedImagesCount]
            if self.measuringType == 'image':
                part_accuracy = self.evaluate_part_accuracy(actual_img, generated_images)
            else:
                part_accuracy = self.evaluate_part_accuracy_in_point(actual_img, generated_images, self.measuringPoint)
            sequence_accuracy.append(part_accuracy)
        return sequence_accuracy

    @staticmethod
    def evaluate_part_accuracy_in_point(actual_images, generated_images, point):
        return PartAccuracyEvaluator.evaluate_part_accuracy_in_point(actual_images, generated_images, point)

    def evaluate_part_accuracy(self, actual_images, generated_images):
        return PartAccuracyEvaluator.evaluate_part_accuracy(actual_images, generated_images, self.errorFunction)

    def evaluate(self):
        overall_accuracy = []
        for sequence in self.imageSequences:
            sequence_accuracy = self.evaluate_sequence(sequence)
            overall_accuracy.append(sequence_accuracy)

        return overall_accuracy

from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.utilities.errorFunctions import trueSkillStatistic
from PIL import Image

class Validation:
    validationSequences = []
    dimension = 64

    def set_validation_data(self, sequences):
        self.validationSequences = sequences
        return self

    def set_dimensions(self, dimension):
        self.dimension = dimension
        return self

    def validate(self, algorithm):
        evaluator = MultiAlgorithmAccuracyEvaluator()
        results = evaluator.set_image_sequences(self.validationSequences) \
            .set_predicted_images_count(8) \
            .set_source_images_count(8) \
            .set_range_step(1) \
            .set_error_function(trueSkillStatistic.TrueSkillStatistic()) \
            .set_measuring_type('image') \
            .set_prediction_algorithms(
            [
                algorithm
            ]
        ).evaluate()

        future_images = 8
        source_images = 4
        dim = self.dimension

        test_images = self.validationSequences[0][0:source_images]
        image_results = algorithm.predict(test_images, future_images)
        sample_image = Image.new('L', (dim*(future_images+source_images), dim*2))

        for i, image in enumerate(test_images):
            sample_image.paste(image, (dim * i, 0))

        for i, image in enumerate(self.validationSequences[0][source_images:source_images+future_images]):
            sample_image.paste(image, (dim * (i + source_images), 0))

        for i, image in enumerate(image_results):
            sample_image.paste(image, (dim * (i + source_images), dim))

        sample_image.show()
        evaluation_processor = EvaluationProcessor()

        averaged_results = []
        for result in results:
            eval_result = evaluation_processor \
                .set_evaluation_results(result) \
                .get_sequence_prediction_averages()[0]
            averaged_results.append(eval_result)
        for i in range(len(averaged_results[0])):
            print("Accuracy " + str(i) + " :" + str(averaged_results[0][i]))
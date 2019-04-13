from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.utilities.errorFunctions import trueSkillStatistic
from PIL import Image

class Validation:
    validationSequences = []
    dimension = 64
    step = 1
    base = 4
    future_images = 16

    def set_future_images(self, images):
        self.future_images = images
        return self

    def set_base(self, base):
        self.base = base
        return self

    def set_step(self, step):
        self.step = step
        return self

    def set_validation_data(self, sequences):
        self.validationSequences = sequences
        return self

    def set_dimensions(self, dimension):
        self.dimension = dimension
        return self

    def validate(self, algorithm):
        evaluator = MultiAlgorithmAccuracyEvaluator()
        source_images = self.step * self.base - self.step + 1
        # results = evaluator.set_image_sequences(self.validationSequences) \
        #     .set_predicted_images_count(self.future_images) \
        #     .set_source_images_count(source_images) \
        #     .set_range_step(4) \
        #     .set_error_function(trueSkillStatistic.TrueSkillStatistic()) \
        #     .set_measuring_type('image') \
        #     .set_prediction_algorithms(
        #     [
        #         algorithm
        #     ]
        # ).evaluate()

        dim = self.dimension

        test_images = self.validationSequences[0][0:source_images:self.step]
        print(self.step, self.base, len(test_images), source_images)
        image_results = algorithm.predict(test_images, self.future_images)
        sample_image = Image.new('L', (dim*(self.future_images+source_images), dim*2))

        for i, image in enumerate(test_images):
            sample_image.paste(image, (dim * i, 0))

        for i, image in enumerate(self.validationSequences[0][source_images:source_images+self.future_images*self.step:self.step]):
            sample_image.paste(image, (dim * (i + self.base), 0))

        for i, image in enumerate(image_results):
            sample_image.paste(image, (dim * (i + self.base), dim))

        sample_image.show()
        # evaluation_processor = EvaluationProcessor()
        #
        # averaged_results = []
        # for result in results:
        #     eval_result = evaluation_processor \
        #         .set_evaluation_results(result) \
        #         .get_sequence_prediction_averages()[0]
        #     averaged_results.append(eval_result)
        # for i in range(len(averaged_results[0])):
        #     print("Accuracy " + str(i) + " :" + str(averaged_results[0][i]))
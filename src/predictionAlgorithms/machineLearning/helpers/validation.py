from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.utilities.errorFunctions import trueSkillStatistic


class Validation:
    validationSequences = []

    def set_validation_data(self, sequences):
        self.validationSequences = sequences
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
        evaluation_processor = EvaluationProcessor()
        averaged_results = []
        for result in results:
            eval_result = evaluation_processor \
                .set_evaluation_results(result) \
                .get_sequence_prediction_averages()[0]
            averaged_results.append(eval_result)
        for i in range(len(averaged_results[0])):
            print("Accuracy " + str(i) + " :" + str(averaged_results[0][i]))
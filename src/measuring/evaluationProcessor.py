import numpy


class EvaluationProcessor:
    evaluationResults = []

    def set_evaluation_results(self, results):
        self.evaluationResults = results
        return self

    @staticmethod
    def get_sequence_prediction_average(sequence_evaluation):
        part_count = len(sequence_evaluation)
        part_sums = numpy.zeros(len(sequence_evaluation[0]))
        for part_evaluation in sequence_evaluation:
            for index in range(len(part_evaluation)):
                part_sums[index] += part_evaluation[index]
        return [x / part_count for x in part_sums]

    def get_sequence_prediction_averages(self):
        averages = []
        for sequence_evaluation in self.evaluationResults:
            average = self.get_sequence_prediction_average(sequence_evaluation)
            averages.append(average)
        return averages

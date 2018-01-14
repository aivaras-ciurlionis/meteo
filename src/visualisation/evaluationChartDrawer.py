from src.measuring.evaluationProcessor import EvaluationProcessor
import plotly.offline as py
import plotly.graph_objs as go


class EvaluationChartDrawer:
    processedResults = None

    def set_evaluation_results(self, evaluation_result):
        evaluation_processor = EvaluationProcessor()
        averages = evaluation_processor\
            .set_evaluation_results(evaluation_result)\
            .get_sequence_prediction_averages()
        self.processedResults = averages[0]
        return self

    def set_processed_results(self, processed_results):
        self.processedResults = processed_results[0]
        return self

    @staticmethod
    def get_x_axis(data_length):
        return ["%d:%02d" % divmod(x*15, 60) for x in range(data_length)]

    def draw_bar_chart(self):
        data_length = len(self.processedResults)
        data = [go.Bar(
            x=self.get_x_axis(data_length),
            y=self.processedResults
        )]
        py.plot(data, filename='bar.html')

    def draw_line_chart(self):
        data_length = len(self.processedResults)
        data = [
            go.Scatter(
                x=self.get_x_axis(data_length),
                y=self.processedResults
            )
        ]
        py.plot(data, filename='line.html')

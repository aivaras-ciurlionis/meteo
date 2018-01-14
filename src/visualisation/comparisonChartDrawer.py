from src.measuring.evaluationProcessor import EvaluationProcessor
import plotly.offline as py
import plotly.graph_objs as go


class ComparisonChartDrawer:
    result1 = None
    result2 = None
    result3 = None

    def set_evaluation_results(self, r1, r2, r3=None):
        evaluation_processor = EvaluationProcessor()

        self.result1 = evaluation_processor\
            .set_evaluation_results(r1)\
            .get_sequence_prediction_averages()[0]

        self.result2 = evaluation_processor \
            .set_evaluation_results(r2) \
            .get_sequence_prediction_averages()[0]

        # self.result3 = evaluation_processor \
        #     .set_evaluation_results(r3) \
        #     .get_sequence_prediction_averages()[0]

        return self

    @staticmethod
    def get_x_axis(data_length):
        return ["%d:%02d" % divmod((x+1)*15, 60) for x in range(data_length)]

    def draw_line_chart(self):
        data_length = len(self.result1)
        data = [
            go.Scatter(
                x=self.get_x_axis(data_length),
                y=self.result1,
                name='Persistency'
            ),
            go.Scatter(
                x=self.get_x_axis(data_length),
                y=self.result2,
                name='XY'
            )
        ]
        py.plot(data, filename='line.html')

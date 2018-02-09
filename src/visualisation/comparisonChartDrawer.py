from src.measuring.evaluationProcessor import EvaluationProcessor
import plotly.offline as py
import plotly.graph_objs as go


class ComparisonChartDrawer:
    results = []
    names = []

    def set_names(self, names):
        self.names = names
        return self

    def set_evaluation_results(self, results):
        evaluation_processor = EvaluationProcessor()
        for result in results:
            eval_result = evaluation_processor\
             .set_evaluation_results(result)\
             .get_sequence_prediction_averages()[0]
            self.results.append(eval_result)
        return self

    @staticmethod
    def get_x_axis(data_length):
        return ["%d:%02d" % divmod((x+1)*15, 60) for x in range(data_length)]

    def draw_line_chart(self):
        data_length = len(self.results[0])
        data = []
        for i, result in enumerate(self.results):
            data.append(
                go.Scatter(
                    x=self.get_x_axis(data_length),
                    y=result,
                    name=self.names[i]
                )
            )
        py.plot(data, filename='line.html')

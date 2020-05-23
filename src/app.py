from flask import Flask, request

from src.measuring.postProcessMeasuring import PostProcessMeasuring
from src.prediction.predictionWrapper import PredictionWrapper

from src.utilities.errorFunctions import imagesMeanSquareError
from src.utilities.errorFunctions import trueSkillStatistic

app = Flask(__name__)

import platform
print(platform.python_version())


@app.route("/last-prediction")
def last_prediction():
    with open('last-prediction.json', 'r') as content_file:
        content = content_file.read()
        return content


@app.route("/predict")
def predict():
    print('Prediction Start')
    result = PredictionWrapper.predict()
    return result


@app.route("/predict-historical")
def predict_historical():
    date = request.args['date']
    result = PredictionWrapper.predict(date)
    return result


@app.route("/accuracy")
def get_accuracy():
    files = request.args.getlist('files')
    error_fun = request.args['error']

    measuring = PostProcessMeasuring().set_files(files)
    if error_fun == 'mse':
        measuring.set_error_function(imagesMeanSquareError.ImagesMeanSquareError())
    if error_fun == 'hk':
        measuring.set_error_function(trueSkillStatistic.TrueSkillStatistic())
    return measuring.evaluate()
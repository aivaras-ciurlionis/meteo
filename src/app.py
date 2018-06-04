from flask import Flask

from src.prediction.predictionWrapper import PredictionWrapper

app = Flask(__name__)


@app.route("/last-prediction")
def last_prediction():
    with open('last-prediction.json', 'r') as content_file:
        content = content_file.read()
        return content


@app.route("/predict")
def predict():
    result = PredictionWrapper.predict()
    return result


if __name__ == "__main__":
    app.run()
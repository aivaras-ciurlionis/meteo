from src.predictionAlgorithms.baseAlgorithm import BaseAlgorithm


class PersistencyAlgorithm(BaseAlgorithm):
    name = 'Persistency'

    def predict(self, source_images, count):
        predictions = []
        for index in range(count):
            predictions.append(source_images[-1])
        return predictions

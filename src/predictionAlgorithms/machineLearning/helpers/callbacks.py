import keras
from sklearn.metrics import roc_auc_score

from src.predictionAlgorithms.machineLearning.helpers.validation import Validation


class Callbacks(keras.callbacks.Callback):
    validationSequences = []
    algorithm = None
    number = 1
    validation_frequency = 1
    size = 64
    step = 1
    base = 4

    def set_step(self, step):
        self.step = step
        return self

    def set_base(self, base):
        self.base = base
        return base

    def set_size(self, size):
        self.size = size
        return self

    def set_validation_frequency(self, frequency):
        self.validation_frequency = frequency
        return self

    def set_validation_data(self, validation_data):
        self.validationSequences = validation_data
        return self

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        return self

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        print('epoch '+str(self.number))
        if self.number % self.validation_frequency != 0:
            self.number += 1
            return
        validation = Validation()
        validation.set_validation_data(self.validationSequences)\
            .set_dimensions(self.size)\
            .set_base(self.base)\
            .set_step(self.step)\
            .validate(self.algorithm)
        self.losses.append(logs.get('loss'))
        self.number += 1
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

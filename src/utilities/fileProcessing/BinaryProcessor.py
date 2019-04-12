import numpy as np


class BinaryProcessor:

    @staticmethod
    def save_data(data, date):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)

        np.save('../binary_images/x-' + date + str(x.shape), x)
        np.save('../binary_images/y-' + date + str(x.shape), y)
from math import sqrt
from sklearn.metrics import mean_squared_error


class ImagesMeanSquareError:

    @staticmethod
    def get_mean_square_error(image1data, image2data):
        return sqrt(mean_squared_error(image1data, image2data))

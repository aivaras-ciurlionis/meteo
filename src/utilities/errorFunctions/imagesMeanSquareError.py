from math import sqrt
from sklearn.metrics import mean_squared_error


class ImagesMeanSquareError:

    def get_error(self, image1data, image2data):
        return sqrt(mean_squared_error(image1data, image2data))

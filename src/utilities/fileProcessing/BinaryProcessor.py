import numpy as np
import os

class BinaryProcessor:

    @staticmethod
    def save_data(data, date):
        x, y = data
        size = 100
        count = int(np.ceil(len(x) / size))
        os.mkdir('../binary_images/' + date)
        for i in range(count):
            sliceX = x[i*size:i*size + size]
            sliceY = y[i*size:i*size + size]
            x = np.asarray(sliceX)
            y = np.asarray(sliceY)
            print('start save X ' + str(i) + 'of ' + str(count))
            np.save('../binary_images/' + date + '/x-' + date + str(x.shape) + '-' + str(i), x)
            print('start save Y'  + str(i) + 'of ' + str(count))
            np.save('../binary_images/' + date + '/y-' + date + str(y.shape) + '-' + str(i), y)
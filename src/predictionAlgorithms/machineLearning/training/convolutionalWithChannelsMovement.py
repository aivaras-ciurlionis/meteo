from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD
import numpy as np
import os

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class ConvolutionalWithChannelsMovement:

    @staticmethod
    def train(data, size, channels):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)
        model = Sequential([
            Conv2D(filters=5,
                   kernel_size=(5, 5),
                   activation='relu',
                   data_format='channels_first',
                   input_shape=(channels, size, size)),
            Conv2D(filters=1,
                   kernel_size=(5, 5),
                   activation='relu',
                   data_format='channels_first')
        ])

        model.compile(
            optimizer=SGD(lr=0.005),
            loss='mse'
        )

        model.fit(x, y, epochs=50)

        f = model.predict(np.asarray([x[0]]))
        print(np.max(x[0][0]))
        print(np.max(y[0][0]))
        print(np.max(f[0][0]))


        src = os.path.join('../output', 'test.png')
        new_im = Image.new("L", (56, 56))
        data = f[0][0].flatten()
        data = np.array(list(map(lambda x: int(x * 16), data)))

        model.save('conv_chan_movement_model.h5')

        new_im.putdata(data)
        new_im.save(src)

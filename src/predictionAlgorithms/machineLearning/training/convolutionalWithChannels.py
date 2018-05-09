from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD
import numpy as np


class ConvolutionalWithChannels:

    @staticmethod
    def train(data, size, channels):
        x, y = data
        x = np.asarray(x)
        y = np.asarray(y)
        model = Sequential([
            Conv2D(filters=1,
                   kernel_size=(3, 3),
                   activation='sigmoid',
                   data_format='channels_first',
                   input_shape=(channels, size, size))
        ])

        model.compile(
            optimizer=SGD(lr=0.5),
            loss='mse'
        )

        model.fit(x, y, epochs=40)

        #model.save('conv_chan_model.h5')

        f = model.predict(np.asarray([x[0]]))
        print(f[0])
        print(y[0])

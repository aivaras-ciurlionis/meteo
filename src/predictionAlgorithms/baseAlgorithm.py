import numpy as np
from keras.models import model_from_json

IMAGES_PER_EXAMPLE = 4


class BaseAlgorithm:
    name = 'Base'
    step = 1
    size = 64
    base = 4

    def predict(self, source_images, count):
        return source_images

    def with_step(self, step):
        self.step = step
        return self

    def with_size(self, size):
        self.size = size
        return self

    def with_base(self, base):
        self.base = base
        return self

    def get_random_layer(self, i):
        seeds = [1873, 2764, 1236]
        np.random.seed(seeds[i])
        return np.random.rand(self.size, self.size) * 255

    def get_elevation_map(self):
        elevation_map = None
        try:
            elevation_map = np.load('/app/src/savedModels/elevation.npy')
        except:
            elevation_map = np.load('savedModels/elevation.npy')
        return elevation_map

    def get_model_input(self, temp, elevation_layers, random_layers, isConvLstm=False):
        if isConvLstm:
            return temp.reshape((-1, IMAGES_PER_EXAMPLE, 1, self.size, self.size))
        has_additional_layer = elevation_layers > 0 or random_layers > 0
        offset_to_add = random_layers + elevation_layers
        temp_expanded = np.zeros((1, IMAGES_PER_EXAMPLE + offset_to_add, self.size, self.size))
        temp_expanded[:, :IMAGES_PER_EXAMPLE, :, :] = temp
        if has_additional_layer:
            for i in range(elevation_layers):
                temp_expanded[:, IMAGES_PER_EXAMPLE + i, :, :] = self.get_elevation_map()
            for i in range(random_layers):
                temp_expanded[:, IMAGES_PER_EXAMPLE + elevation_layers + i, :, :] = self.get_random_layer(i)
        return temp_expanded

    def load_ml_model(self, filename):
        print('loading model', filename)
        json_file = open(filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(filename + ".h5")
        return model

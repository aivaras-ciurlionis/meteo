import numpy as np


class SequenceProcessor:
    sequence = []

    def set_sequence(self, sequence):
        self.sequence = sequence
        return self

    def merge_to_channels(self, channel_count):
        if len(self.sequence) < channel_count:
            return []
        results_x = []
        results_y = []
        for i in range(0, len(self.sequence) - channel_count - 1):
            x_images = self.sequence[i:i + channel_count]
            y_image_data =np.asarray(self.sequence[i + channel_count + 1])[1:63, 1:63]
            y_image = [np.array(list(map(lambda x: x / 255, y_image_data)))]
            merged = self.merge_images(x_images)
            results_x.append(merged)
            results_y.append(y_image)

        return results_x, results_y

    def merge_images(self, images):
        merged = []
        for image in images:
            data = np.asarray(image)
            data = np.array(list(map(lambda x: x / 255, data)))
            merged.append(data)
        return merged

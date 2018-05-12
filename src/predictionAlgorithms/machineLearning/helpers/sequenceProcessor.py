import numpy as np

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class SequenceProcessor:
    sequence = []

    def set_sequence(self, sequence):
        self.sequence = sequence
        return self

    def merge_to_channels(self, channel_count):
        converter = PixelsRainStrengthConverter()
        if len(self.sequence) < channel_count:
            return []
        results_x = []
        results_y = []
        for i in range(0, len(self.sequence) - channel_count - 1):
            images = self.sequence[i:i + channel_count + 1]
            converted_images = converter.convert_loaded(images)
            x_images = converted_images[0:channel_count]
            print('mx', np.max(x_images[0]))

            y_image_data = [np.asarray(converted_images[-1])[4:60, 4:60]]
            merged = self.merge_images(x_images)
            results_x.append(merged)
            results_y.append(y_image_data)


        return results_x, results_y

    def merge_images(self, images):
        merged = []
        for image in images:
            data = np.asarray(image)
            merged.append(data)
        return merged

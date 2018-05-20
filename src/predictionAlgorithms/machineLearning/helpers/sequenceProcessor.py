import numpy as np

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


class SequenceProcessor:
    sequences = []

    def set_sequence(self, sequence):
        self.sequence = sequence
        return self

    def set_sequences(self, sequence):
        self.sequences = sequence
        return self

    def merge_sequences_to_channels(self, channel_count, result_count=1):
        results_x = []
        results_y = []
        for sequence in self.sequences:
            sequence_result = self.merge_to_channels(channel_count, sequence, result_count)
            if len(sequence_result) > 1:
                results_x += sequence_result[0]
                results_y += sequence_result[1]
        return results_x, results_y

    def merge_to_channels(self, channel_count, sequence, result_count=1):
        converter = PixelsRainStrengthConverter()
        if len(sequence) < channel_count + result_count:
            return []
        results_x = []
        results_y = []
        for i in range(0, len(sequence) - channel_count - result_count - 1):
            images = sequence[i:i + channel_count + result_count]
            converted_images = converter.convert_loaded(images)
            x_images = converted_images[0:channel_count]
            y_images = converted_images[-result_count:]
            y_images_data = []

            for y_image in y_images:
                y_image_data = np.asarray(y_image)[0:64, 0:64]
                y_images_data.append(y_image_data)

            if len(y_images_data) < 2:
                y_images_data = np.asarray([y_images_data[0]])

            merged_x = self.merge_images(x_images)
            merged_y = self.merge_images(y_images_data)
            results_x.append(merged_x)
            results_y.append(merged_y)
        return results_x, results_y

    def merge_images(self, images):
        merged = []
        for image in images:
            data = np.asarray(image)
            merged.append(data)
        return merged

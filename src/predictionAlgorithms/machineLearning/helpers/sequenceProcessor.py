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

    def merge_sequences_to_channels(self, channel_count):
        results_x = []
        results_y = []
        for sequence in self.sequences:
            sequence_result = self.merge_to_channels(channel_count, sequence)
            if len(sequence_result) > 1:
                results_x += sequence_result[0]
                results_y += sequence_result[1]
        return results_x, results_y

    def merge_to_channels(self, channel_count, sequence):
        converter = PixelsRainStrengthConverter()
        if len(sequence) < channel_count:
            return []
        results_x = []
        results_y = []
        for i in range(0, len(sequence) - channel_count - 1):
            images = sequence[i:i + channel_count + 1]
            converted_images = converter.convert_loaded(images)
            x_images = converted_images[0:channel_count]
            y_image_data = [np.asarray(converted_images[-1])[3:61, 3:61]]
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

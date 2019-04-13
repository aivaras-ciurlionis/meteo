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

    def merge_sequences_to_channels_with_steps(self, channel_count, step, size=64):
        results_x = []
        results_y = []
        for sequence in self.sequences:
            sequence_result = self.merge_to_stepped_channels(channel_count, sequence, size, step)
            if len(sequence_result) > 1:
                results_x += sequence_result[0]
                results_y += sequence_result[1]
        return results_x, results_y

    def merge_sequences_to_channels(self, channel_count, result_count=1, size=64):
        results_x = []
        results_y = []
        for sequence in self.sequences:
            sequence_result = self.merge_to_channels(channel_count, sequence, size, result_count)
            if len(sequence_result) > 1:
                results_x += sequence_result[0]
                results_y += sequence_result[1]
        return results_x, results_y

    def has_rain_treshold(self, images, treshold):
        for image in images:
            count = np.count_nonzero(image)
            if count < treshold:
                return False
        return True

    def merge_to_stepped_channels(self, channel_count, sequence, size, step=1):
        converter = PixelsRainStrengthConverter()
        if len(sequence) < (channel_count + 1) * step:
            return []
        results_x = []
        results_y = []
        for i in range(0, len(sequence) - (channel_count + 1) * step - 1):
            images = sequence[i:i + (channel_count + 1) * step]
            converted_images = converter.convert_loaded(images)
            x_images = converted_images[0:(step*(channel_count-1))+1:step]
            y_image = converted_images[(step*(channel_count-1))+step]
            y_images_data = np.asarray([y_image])
            merged_x = self.merge_images(x_images)
            merged_y = self.merge_images(y_images_data)
            if self.has_rain_treshold(merged_x, size) and self.has_rain_treshold(merged_y, size / 10):
                results_x.append(merged_x)
                results_y.append(merged_y)
        return results_x, results_y

    def merge_to_channels(self, channel_count, sequence, size, result_count=1):
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
                y_image_data = np.asarray(y_image)[0:size, 0:size]
                y_images_data.append(y_image_data)

            if len(y_images_data) < 2:
                y_images_data = np.asarray([y_images_data[0]])

            merged_x = self.merge_images(x_images)
            merged_y = self.merge_images(y_images_data)
            if self.has_rain_treshold(merged_x, size) and self.has_rain_treshold(merged_y, size / 10):
                results_x.append(merged_x)
                results_y.append(merged_y)
        return results_x, results_y

    def merge_images(self, images):
        merged = []
        for image in images:
            data = np.asarray(image)
            merged.append(data)
        return merged

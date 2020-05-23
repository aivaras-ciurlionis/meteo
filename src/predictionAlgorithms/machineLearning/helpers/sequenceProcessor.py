import numpy as np

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
import os

class SequenceProcessor:
    sequences = []
    save_n = 0

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

    def merge_sequences_to_channels(self, channel_count, result_count=1, size=64, name='results', timed=False):
        results_x = []
        results_y = []
        i = 0
        c = len(self.sequences)
        if not os.path.exists('../binary_images/' + name):
            os.makedirs('../binary_images/' + name)
        for sequence in self.sequences:
            sequence_result = self.merge_to_channels(channel_count, sequence, size, result_count, timed)
            if len(sequence_result) > 1:
                results_x += sequence_result[0]
                results_y += sequence_result[1]
                results_x, results_y = self.save_and_remove(results_x, results_y, name)
            print(i / c, '%')
            i += 1
        return results_x, results_y

    def save_and_remove(self, x_data, y_data, name):
        size = 1000
        if len(x_data) < size:
            return x_data, y_data
        count = int(np.ceil(len(x_data) / size))
        for i in range(count):
            sliceX = x_data[i*size:i*size + size]
            sliceY = y_data[i*size:i*size + size]
            if len(sliceX) < size:
                return sliceX, sliceY
            x = np.asarray(sliceX)
            y = np.asarray(sliceY)
            print('start save X ' + str(i) + 'of ' + str(count))
            np.save('../binary_images/' + name + '/x-' + name + str(x.shape) + '-' + str(self.save_n), x)
            print('start save Y'  + str(i) + 'of ' + str(count))
            np.save('../binary_images/' + name + '/y-' + name + str(y.shape) + '-' + str(self.save_n), y)
            self.save_n += 1

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

    def merge_to_channels(self, channel_count, sequence, size, result_count=1, timed=False):
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

            merged_x = self.merge_images(x_images, timed)
            merged_y = self.merge_images(y_images_data, timed and result_count > 1)
            if self.has_rain_treshold(merged_x, size) and self.has_rain_treshold(merged_y, size / 10):
                results_x.append(merged_x)
                results_y.append(merged_y)
        return results_x, results_y

    def merge_images(self, images, timed=False):
        merged = []
        for image in images:
            data = np.asarray(image)
            if timed:
                data = np.asarray([data])
            merged.append(data)
        return merged

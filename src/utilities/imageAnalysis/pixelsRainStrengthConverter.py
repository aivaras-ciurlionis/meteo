import numpy
from PIL import Image
from src.utilities.imageAnalysis.pixelToRainStrengthConverter import PixelToRainStrengthConverter


class PixelsRainStrengthConverter:

    @staticmethod
    def convert_images(images):
        image_matrices = []
        for image in images:
            image_matrices.append(PixelsRainStrengthConverter.convert_image_to_strength_image(image))
        return image_matrices

    @staticmethod
    def convert_image_to_matrix(image):
        converter = PixelToRainStrengthConverter()
        image_matrix = numpy.zeros(image.size)
        pixels = image.getdata()
        i = 0
        for pixel in pixels:
            strength = converter.convert_to_strength(pixel)
            coordinates = numpy.unravel_index(i, image.size)
            image_matrix[coordinates[0]][coordinates[1]] = strength
            i += 1
        image.close()
        return image_matrix

    @staticmethod
    def convert_image_to_strength_image(image):
        converter = PixelToRainStrengthConverter()
        matrix = PixelsRainStrengthConverter.convert_image_to_matrix(image)
        strength_list = matrix.flatten()
        strength_list = list(map(lambda s: converter.convert_to_gray(s), strength_list))
        converted_image = Image.new('L', image.size)
        converted_image.putdata(strength_list)
        image.close()
        return converted_image

    @staticmethod
    def normalise_image(image):
        data = image.getdata()
        strength_list = list(map(lambda p: p / 255, data))
        #image.close()
        return strength_list

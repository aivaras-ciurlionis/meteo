import numpy
from PIL import Image
from src.utilities.imageAnalysis.pixelToRainStrengthConverter import PixelToRainStrengthConverter


class PixelsRainStrengthConverter:

    @staticmethod
    def convert_images(images, to_categories=False):
        image_matrices = []
        for image in images:
            if to_categories:
                image_matrices.append(PixelsRainStrengthConverter.to_categories(image))
            else:
                image_matrices.append(PixelsRainStrengthConverter.convert_image_to_strength_image(image))
        return image_matrices

    @staticmethod
    def to_categories(image):
        converter = PixelToRainStrengthConverter()
        pixels = image.getdata()
        categories = []
        for pixel in pixels:
            strength = converter.convert_to_strength(pixel)
            categories.append(strength)
        categories = numpy.asarray(categories)
        categories = numpy.reshape(categories, image.size)
        return categories

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
    def convert_gray_strength_to_source(image):
        converter = PixelToRainStrengthConverter()
        data = image.copy().getdata()
        strength_list = list(map(lambda p: int(p/16), data))
        pixels = list(map(lambda s: converter.convert_to_pixel(s), strength_list))
        converted_image = Image.new('RGBA', image.size)
        converted_image.putdata(pixels)
        return converted_image

    @staticmethod
    def normalise_image(image):
        s = image.size[0]
        crop_amount = int(s*0.10)
        data = image.copy().crop((crop_amount, crop_amount, s-crop_amount, s-crop_amount)).getdata()
        strength_list = list(map(lambda p: p / 255, data))
        return strength_list

    @staticmethod
    def get_normalised_accuracy_in(image, point):
        return image.getpixel(point) / 255

    @staticmethod
    def convert_loaded(images):
        data = []
        for image in images:
            image_data = image.getdata()
            image_data = list(map(lambda d: int(d/16), image_data))
            categories = numpy.asarray(image_data)
            categories = numpy.reshape(categories, image.size)
            data.append(categories)

        return data
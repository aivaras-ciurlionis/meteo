THRESHOLD = 8


def pixel_map(x):
    return {
        1: (0, 60, 255, 191),
        2: (0, 197, 197, 191),
        3: (0, 168, 172, 191),
        4: (0, 129, 97, 191),
        5: (0, 149, 53, 191),
        6: (0, 192, 37, 191),
        7: (0, 232, 9, 191),
        8: (36, 255, 36, 191),
        9: (255, 255, 32, 191),
        10: (255, 230, 0, 191),
        11: (255, 188, 0, 191),
        12: (255, 150, 0, 191),
        13: (255, 94, 0, 191),
        14: (240, 16, 0, 191),
        15: (188, 0, 56, 191),
        16: (255, 0, 255, 191)
    }.get(x, (0, 0, 0, 0))


def is_close_to(pixel, r1, g1, b1):
    return abs(pixel[0]-r1) < THRESHOLD and abs(pixel[1] - g1) < THRESHOLD and abs(pixel[2] - b1) < THRESHOLD


class PixelToRainStrengthConverter:

    @staticmethod
    def strength_to_pixel(strength):
        return pixel_map(strength)


    @staticmethod
    def pixel_to_strength(pixel):
        if is_close_to(pixel, 0, 60, 255):
            return 1
        if is_close_to(pixel, 0, 197, 197):
            return 2
        if is_close_to(pixel, 0, 168, 172):
            return 3
        if is_close_to(pixel, 0, 129, 97):
            return 4
        if is_close_to(pixel, 0, 149, 53):
            return 5
        if is_close_to(pixel, 0, 192, 37):
            return 6
        if is_close_to(pixel, 0, 232, 9):
            return 7
        if is_close_to(pixel, 36, 255, 36):
            return 8
        if is_close_to(pixel, 255, 255, 32):
            return 9
        if is_close_to(pixel, 255, 230, 0):
            return 10
        if is_close_to(pixel, 255, 188, 0):
            return 11
        if is_close_to(pixel, 255, 150, 0):
            return 12
        if is_close_to(pixel, 255, 94, 0):
            return 13
        if is_close_to(pixel, 240, 16, 0):
            return 14
        if is_close_to(pixel, 188, 0, 56):
            return 15
        if is_close_to(pixel, 255, 0, 255):
            return 16
        return 0

    def convert_to_strength(self, image_pixel):
        if image_pixel[0] > 0 or image_pixel[1] > 0 or image_pixel[2] > 0 or image_pixel[3] > 0:
            return self.pixel_to_strength(image_pixel)
        return 0

    def convert_to_pixel(self, strength):
        return self.strength_to_pixel(strength)

    @staticmethod
    def convert_to_gray(strength):
        strength *= 16
        return int(strength)

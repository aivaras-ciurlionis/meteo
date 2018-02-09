from PIL import ImageChops as iC


def offset_fraction(image, x, y, resize_value=1):
    original_size = image.size
    new_size = tuple(resize_value * x for x in image.size)
    working_image = image.resize(new_size)
    working_image = offset(working_image, int(x * resize_value), int(y * resize_value))
    return working_image.resize(original_size)


def offset(image, x, y):
    s = image.size[0]
    i = iC.offset(image, x, y)
    if x > 0:
        i.paste(0, (0, 0, x, s))
    else:
        i.paste(0, (s-x, 0, s, s))

    if y > 0:
        i.paste(0, (0, 0, s, y))
    else:
        i.paste(0, (0, s+y, s, s))
    return i


class Transformations:

    @staticmethod
    def xy_transformation():
        return (
            'XY transformation',
            [
                (lambda image, value: offset(image, value, 0), [-4, 10, 1]),
                (lambda image, value: offset(image, 0, value), [-4, 10, 1]),
            ]
        )

    @staticmethod
    def xy_fraction_transformation():
        return (
            'XY fraction transformation',
            [
                (lambda image, value: offset_fraction(image, value, 0, 4), [-5, 5, .25]),
                (lambda image, value: offset_fraction(image, 0, value, 4), [-5, 5, .25]),
            ]
        )

    @staticmethod
    def rotation():
        return (
            'Rotation',
            [
                (lambda image, value: image.rotate(value), [-10, 10, 1]),
            ]
        )

    @staticmethod
    def xy_rotation():
        return (
            'XY rotation',
            [
                (lambda image, value: offset(image, value, 0), [-5, 5, 1]),
                (lambda image, value: offset(image, 0, value), [-5, 5, 1]),
                (lambda image, value: image.rotate(value), [-5, 5, 1])
            ]
        )

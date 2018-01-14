from src.predictionAlgorithms.correlation.baseTransformationAlgorithm import BaseTransformationAlgorithm
from PIL import ImageChops as iC


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


class XYRotationAlgorithm(BaseTransformationAlgorithm):
    transformations = [
        (lambda image, value: offset(image, value, 0), [-5, 5, 1]),
        (lambda image, value: offset(image, 0, value), [-5, 5, 1]),
        (lambda image, value: image.rotate(value), [-5, 5, 1]),
    ]


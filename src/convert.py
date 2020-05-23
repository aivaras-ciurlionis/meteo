import os
import numpy as np
import time
from PIL import Image
from scipy import sparse
import arrow

THRESHOLD = 2.5
DATE_FORMAT = 'YYYY-MM-DD--HH-mm-ss'
DATE_FORMAT2 = 'YYYY-M-DD--HH-mm-ss'


def is_close_to(pixel, r1, g1, b1):
    return np.abs(pixel[0] - r1) < THRESHOLD and \
           np.abs(pixel[1] - g1) < THRESHOLD and \
           np.abs(pixel[2] - b1) < THRESHOLD

def rgb2gray(rgb):
    return rgb.sum(axis=2)


def pixel_to_strength(pixels):
    pixels[np.abs(pixels - 506.0) < THRESHOLD] = 1
    pixels[np.abs(pixels - 585.0) < THRESHOLD] = 2
    pixels[np.abs(pixels - 531.0) < THRESHOLD] = 3
    pixels[np.abs(pixels - 417.0) < THRESHOLD] = 4
    pixels[np.abs(pixels - 393.0) < THRESHOLD] = 5
    pixels[np.abs(pixels - 420.0) < THRESHOLD] = 6
    pixels[np.abs(pixels - 432.0) < THRESHOLD] = 7
    pixels[np.abs(pixels - 518.0) < THRESHOLD] = 8
    pixels[np.abs(pixels - 733.0) < THRESHOLD] = 9
    pixels[np.abs(pixels - 676.0) < THRESHOLD] = 10
    pixels[np.abs(pixels - 634.0) < THRESHOLD] = 11
    pixels[np.abs(pixels - 596.0) < THRESHOLD] = 12
    pixels[np.abs(pixels - 540.0) < THRESHOLD] = 13
    pixels[np.abs(pixels - 447.0) < THRESHOLD] = 14
    pixels[np.abs(pixels - 435.0) < THRESHOLD] = 15
    pixels[np.abs(pixels - 701.0) < THRESHOLD] = 16
    pixels[pixels > 17] = 0
    return pixels * 16


def to_grayscale(image):
    image.thumbnail((64, 64), Image.NONE)
    pixels = (rgb2gray(np.asarray(image))).flatten()
    converted = pixel_to_strength(pixels)
    return converted



print("XXXXX")
path = "../meteo-data-testing/"
files = os.listdir(path)
files.sort()
count = len(files)
result = np.zeros((count, 64*64))
print(result.shape)
with open('filenames-testing.txt', 'w') as f:
    for i, item in enumerate(files):
        if not os.path.isfile(path + item):
            continue
        f.write("%s\n" % item)
        im = Image.open(path + item)
        result[i] = to_grayscale(im)
        if i % 100 == 0:\
            print((i*100)/count)

print('converting to sparse')
sparse_result = sparse.csr_matrix(result)
print('saving...')
sparse.save_npz('../comb/meteo_64_testing.npz', sparse_result)
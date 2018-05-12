import os

from src.predictionAlgorithms.sequenceCorelation.multiImageSequenceTransformation import \
    MultiImageSequenceTransformation
from src.predictionAlgorithms.transformations import Transformations
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor

from src.utilities.errorFunctions import trueSkillStatistic
from PIL import ImageChops as iC

from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter


def offset(image, x, y):
    s = image.size[0]
    x = int(round(x))
    y = int(round(y))
    i = iC.offset(image, x, y)
    if x > 0:
        i.paste(0, (0, 0, x, s))
    else:
        i.paste(0, (s+x, 0, s, s))

    if y > 0:
        i.paste(0, (0, 0, s, y))
    else:
        i.paste(0, (0, s-y, s, s))
    return i


image_preprocessor = ImagePreprocessor()
sequences = image_preprocessor \
    .set_images_folder('../pics') \
    .set_resized_image_dimension(64) \
    .set_max_images_per_sequence(1000) \
    .set_date_range('2017-10-23 01:15', '2017-10-30 23:00') \
    .load_and_process_images()

n = 4

algorithm = MultiImageSequenceTransformation(Transformations.xy_transformation(),
                                             trueSkillStatistic.TrueSkillStatistic(),
                                             n)

for sn, sequence in enumerate(sequences):
    for i in range(0, len(sequence) - n):
        source_images = sequence[i:i + n]
        result_image = sequence[i + n + 1]
        movement_vector = algorithm.find_best_movement_vector_multi(source_images[-n].copy(), source_images[-n+1:])
        print(movement_vector)
        source_images.append(result_image)
        for j, img in enumerate(source_images):
            next_img = offset(img, -movement_vector[0]*j, -movement_vector[1]*j)
            src = os.path.join('../channels_input', str(sn) + '-' + str(i) + '-' + str(j) + '.png')
            converted_image = PixelsRainStrengthConverter.convert_gray_strength_to_source(next_img)
            converted_image.save(src)

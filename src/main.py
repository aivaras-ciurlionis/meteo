from src.utilities.fileProcessing.ImageSequencesLoader import ImageSequencesLoader
from src.utilities.fileProcessing.ImageLoader import ImageLoader
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer
from src.utilities.imageAnalysis.imagesMeanSquareError import ImagesMeanSquareError

sequencesLoader = ImageSequencesLoader()
imageLoader = ImageLoader()
imageResizer = ImageResizer()
converter = PixelsRainStrengthConverter()
mse = ImagesMeanSquareError()

imageSequences = sequencesLoader.select_folder('../pics').load_sequences()
images = imageLoader.set_image_folder('../pics').set_sequence(imageSequences[0]).load_sequence_images()
test_images = images[0:10]
imageResizer.set_images(test_images).resize_images((256, 256))
converted_images = converter.convert_images(test_images)

normalisedImage1 = converter.normalise_image(converted_images[0])
for i in range(1, 10):
    normalisedImage = converter.normalise_image(converted_images[i])
    print(mse.get_mean_square_error(normalisedImage1, normalisedImage))
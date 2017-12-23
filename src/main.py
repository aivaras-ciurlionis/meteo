from src.utilities.fileProcessing.ImageSequencesLoader import ImageSequencesLoader
from src.utilities.fileProcessing.ImageLoader import ImageLoader
from src.utilities.imageAnalysis.pixelsRainStrengthConverter import PixelsRainStrengthConverter
from src.utilities.imageProcessing.imageResizer import ImageResizer

sequencesLoader = ImageSequencesLoader()
imageLoader = ImageLoader()
imageResizer = ImageResizer()
converter = PixelsRainStrengthConverter()

imageSequences = sequencesLoader.select_folder('../pics').load_sequences()
images = imageLoader.set_image_folder('../pics').set_sequence(imageSequences[0]).load_sequence_images()
test_images = images[0:1]
imageResizer.set_images(test_images).resize_images((256, 256))
converted_images = converter.convert_images(test_images)
converted_images[0].show()
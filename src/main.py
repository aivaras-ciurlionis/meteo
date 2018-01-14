from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor

image_preprocessor = ImagePreprocessor()
evaluator = AccuracyEvaluator()

sequences = image_preprocessor\
    .set_images_folder('../pics')\
    .set_resized_image_dimension(128)\
    .set_max_images_per_sequence(50)\
    .load_and_process_images()

evaluation_result = evaluator\
    .set_image_sequences(sequences)\
    .set_predicted_images_count(8)\
    .set_prediction_algorithm(PersistencyAlgorithm())\
    .evaluate()

print(evaluation_result[0][1])
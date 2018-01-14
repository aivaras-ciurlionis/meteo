from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.measuring.evaluationProcessor import EvaluationProcessor
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.visualisation.evaluationChartDrawer import EvaluationChartDrawer

image_preprocessor = ImagePreprocessor()
evaluator = AccuracyEvaluator()
evaluation_processor = EvaluationProcessor()

sequences = image_preprocessor\
    .set_images_folder('../pics')\
    .set_resized_image_dimension(32)\
    .set_max_images_per_sequence(1000)\
    .set_date_range('2017-10-28 03:00', '2017-11-01 00:00')\
    .load_and_process_images()

evaluation_result = evaluator\
    .set_image_sequences(sequences)\
    .set_predicted_images_count(40)\
    .set_prediction_algorithm(PersistencyAlgorithm())\
    .evaluate()

drawer = EvaluationChartDrawer()
drawer.set_evaluation_results(evaluation_result).draw_line_chart()
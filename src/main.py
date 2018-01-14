from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.measuring.evaluationProcessor import EvaluationProcessor
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.predictionAlgorithms.correlation.xyRotationAlgorithm import XYRotationAlgorithm
from src.predictionAlgorithms.correlation.xyTransformationAlgorithm import XYTransformationAlgorithm
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.visualisation.comparisonChartDrawer import ComparisonChartDrawer
from src.visualisation.evaluationChartDrawer import EvaluationChartDrawer
from PIL import ImageChops as iC

image_preprocessor = ImagePreprocessor()
evaluator = AccuracyEvaluator()
evaluation_processor = EvaluationProcessor()

sequences = image_preprocessor\
    .set_images_folder('../pics')\
    .set_resized_image_dimension(64)\
    .set_max_images_per_sequence(500)\
    .set_date_range('2017-10-23 01:15', '2017-10-30 23:00')\
    .load_and_process_images()

# xyTransform = XYRotationAlgorithm()
# first_images = sequences[0][0:5]
# images = xyTransform.predict(first_images, 50)
# for i, image in enumerate(images):
#     image.save('../output/image'+str(i)+'.png')

persistency_result = evaluator\
    .set_image_sequences(sequences)\
    .set_predicted_images_count(10)\
    .set_source_images_count(2)\
    .set_prediction_algorithm(PersistencyAlgorithm())\
    .evaluate()

XY_result = evaluator\
    .set_image_sequences(sequences)\
    .set_predicted_images_count(10)\
    .set_source_images_count(2)\
    .set_prediction_algorithm(XYTransformationAlgorithm())\
    .evaluate()

drawer = ComparisonChartDrawer()
drawer\
    .set_evaluation_results(persistency_result, XY_result)\
    .draw_line_chart()


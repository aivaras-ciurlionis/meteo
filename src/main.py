from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.prediction.multiAlgorithmPrediction import MultiAlgorithmPrediction
from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.predictionAlgorithms.fractionCorelation.multiImageStepTransformation import MultiImageStepTransformation

from src.predictionAlgorithms.transformations import Transformations
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.visualisation.comparisonChartDrawer import ComparisonChartDrawer
from src.visualisation.evaluationChartDrawer import EvaluationChartDrawer
from PIL import ImageChops as iC

image_preprocessor = ImagePreprocessor()
evaluator = MultiAlgorithmAccuracyEvaluator()
evaluation_processor = EvaluationProcessor()

# sequences = image_preprocessor\
#     .set_images_folder('../pics')\
#     .set_resized_image_dimension(200)\
#     .set_max_images_per_sequence(600)\
#     .set_date_range('2017-10-23 01:15', '2017-10-30 23:00')\
#     .load_and_process_images()
#
# result = evaluator\
#     .set_image_sequences(sequences)\
#     .set_predicted_images_count(30)\
#     .set_source_images_count(6)\
#     .set_measuring_point((32,32))\
#     .set_range_step(40)\
#     .set_measuring_type('image')\
#     .set_prediction_algorithms(
#     [
#         PersistencyAlgorithm(),
#         XYTransformationAlgorithm(),
#         XYFractionTransformationAlgorithm()
#     ]
#     )\
#     .evaluate()
#
# drawer = ComparisonChartDrawer()
# drawer\
#     .set_evaluation_results(result)\
#     .set_names(['Persistency', 'XY Transformation', 'XY 4 average Transformation'])\
#     .draw_line_chart()

prediction = MultiAlgorithmPrediction()
prediction.set_images_folder('../pics')\
    .set_output_dir('../../meteo-angular/src/assets/images')\
    .set_predicted_images(10)\
    .set_resize_size(150)\
    .set_source_date('2017-10-25 17:45')\
    .set_algorithms(
    [
        PersistencyAlgorithm(),
        BaseTransformation(Transformations.xy_transformation()),
        BaseTransformation(Transformations.xy_rotation()),
        BaseTransformation(Transformations.rotation()),

        MultiImageStepTransformation(Transformations.xy_transformation()),
        MultiImageStepTransformation(Transformations.xy_rotation()),
        MultiImageStepTransformation(Transformations.rotation())

    ]
    )\
    .predict()\
    .dump_to_json('../../meteo-angular/src/assets')

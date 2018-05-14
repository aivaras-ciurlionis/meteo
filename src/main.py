from src.measuring.accuracyEvaluator import AccuracyEvaluator
from src.measuring.evaluationProcessor import EvaluationProcessor
from src.measuring.multiAlgorithmAccuracyEvaluator import MultiAlgorithmAccuracyEvaluator
from src.prediction.multiAlgorithmPrediction import MultiAlgorithmPrediction
from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.predictionAlgorithms.fractionCorelation.multiImageStepTransformation import MultiImageStepTransformation
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsAlgorithm import \
    ConvolutionalChannelsAlgorithm
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.sequenceCorelation.multiImageSequenceTransformation import \
    MultiImageSequenceTransformation

from src.predictionAlgorithms.transformations import Transformations
from src.utilities.fileProcessing.ImagePreprocessor import ImagePreprocessor
from src.visualisation.comparisonChartDrawer import ComparisonChartDrawer
from src.visualisation.evaluationChartDrawer import EvaluationChartDrawer
from PIL import ImageChops as iC

from src.utilities.errorFunctions import imagesMeanSquareError
from src.utilities.errorFunctions import trueSkillStatistic



import src.predictionAlgorithms.machineLearning.algorithms

# image_preprocessor = ImagePreprocessor()
# evaluator = MultiAlgorithmAccuracyEvaluator()
# evaluation_processor = EvaluationProcessor()
#
# sequences = image_preprocessor\
#     .set_images_folder('../pics')\
#     .set_resized_image_dimension(64)\
#     .set_max_images_per_sequence(500)\
#     .set_date_range('2017-10-28 03:30', '2017-10-30 23:00')\
#     .load_and_process_images()
#
# result = evaluator\
#     .set_image_sequences(sequences)\
#     .set_predicted_images_count(10)\
#     .set_source_images_count(4)\
#     .set_measuring_point((32,32))\
#     .set_range_step(1)\
#     .set_error_function(trueSkillStatistic.TrueSkillStatistic())\
#     .set_measuring_type('image')\
#     .set_prediction_algorithms(
#     [
#         PersistencyAlgorithm(),
#         MultiImageSequenceTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 4),
#         ConvolutionalChannelsAlgorithm(),
#         ConvolutionalChannelsMovementAlgorithm()
#     ]
#     )\
#     .evaluate()
#
# drawer = ComparisonChartDrawer()
# drawer\
#     .set_evaluation_results(result)\
#     .set_names(['Persistency', 'XY Sequence', 'CNN', 'CNN move'])\
#     .draw_line_chart()
#


prediction = MultiAlgorithmPrediction()
prediction.set_images_folder('../pics')\
    .set_output_dir('../../meteo-angular/src/assets/images')\
    .set_predicted_images(10)\
    .set_resize_size(64)\
    .set_error_function(trueSkillStatistic.TrueSkillStatistic())\
    .set_source_date('2017-10-25 16:15')\
    .set_algorithm_names([
        'CNN channels',
        'CNN channels movement',
        'Persistency',
        'Basic transformation',
        'Step transformation',
        'Sequence transformation'
    ])\
    .set_algorithms(
    [
        ConvolutionalChannelsAlgorithm(),
        ConvolutionalChannelsMovementAlgorithm(),
        PersistencyAlgorithm(),
        BaseTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic()),
        MultiImageStepTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 3),
        MultiImageSequenceTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 4)
    ]
    )\
    .predict()\
    .dump_to_json('../../meteo-angular/src/assets')

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
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalMovementMulti8Algorithm import \
    ConvolutionalMovementMulti8Algorithm
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalMovementMultiAlgorithm import \
    ConvolutionalMovementMultiAlgorithm
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
#     .set_date_range('2017-10-27 00:00', '2017-11-01 00:00')\
#     .load_and_process_images()
#
# result = evaluator\
#     .set_image_sequences(sequences)\
#     .set_predicted_images_count(8)\
#     .set_source_images_count(8)\
#     .set_measuring_point((32,32))\
#     .set_range_step(1)\
#     .set_error_function(trueSkillStatistic.TrueSkillStatistic())\
#     .set_measuring_type('image')\
#     .set_prediction_algorithms(
#     [
#         PersistencyAlgorithm(),
#         BaseTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic()),
#         MultiImageStepTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 4),
#         MultiImageSequenceTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 4),
#         ConvolutionalChannelsMovementAlgorithm()
#     ]
#     )\
#     .evaluate()
#
# drawer = ComparisonChartDrawer()
# drawer\
#     .set_evaluation_results(result)\
#     .set_names(['Persistency', 'Base transform', 'Step transform', 'Sequence', 'CNN movement'])\
#     .draw_line_chart()



prediction = MultiAlgorithmPrediction()
prediction.set_images_folder('../pics-full/MeteoData/Data')\
    .set_output_dir('../../meteo-angular/src/assets/images')\
    .set_predicted_images(16)\
    .set_resize_size(128)\
    .set_error_function(trueSkillStatistic.TrueSkillStatistic())\
    .set_source_date('2018-10-02 21:00')\
    .set_algorithm_names([
        'CNN-based',
        'Sequence translation'
    ])\
    .set_algorithms(
    [
        ConvolutionalChannelsMovementAlgorithm(),
        MultiImageSequenceTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic(), 4)
    ]
    )\
    .predict()\
    .dump_to_json('../../meteo-angular/src/assets')

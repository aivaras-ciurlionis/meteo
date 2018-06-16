import json

import os

from src.meteoApi.meteoDataDownloader import MeteoDataDownloader
from src.prediction.imagesPrediction import ImagesPrediction
from src.predictionAlgorithms.correlation.baseTransformation import BaseTransformation
from src.predictionAlgorithms.correlation.persistencyAlgorithm import PersistencyAlgorithm
from src.predictionAlgorithms.fractionCorelation.multiImageStepTransformation import MultiImageStepTransformation
from src.predictionAlgorithms.machineLearning.algorithms.ConvolutionalChannelsMovementAlgorithm import \
    ConvolutionalChannelsMovementAlgorithm
from src.predictionAlgorithms.sequenceCorelation.multiImageSequenceTransformation import \
    MultiImageSequenceTransformation
from src.predictionAlgorithms.transformations import Transformations
from src.storage.blobUploader import BlobUploader
from src.utilities.errorFunctions import trueSkillStatistic


class PredictionWrapper:

    @staticmethod
    def predict(date=None):
        meteo = MeteoDataDownloader()
        prediction = ImagesPrediction()
        uploader = BlobUploader()
        source_time = date
        result = None
        if date is None:
            result = meteo.set_base_dir('../meteo-out').set_images_count(10).load_radar_data()
            source_time = result['source_time']
        r = prediction \
            .set_resize_size(64) \
            .set_source_date(source_time) \
            .set_predicted_images(16) \
            .set_output_dir('../output') \
            .set_images_folder('../meteo-out/actual') \
            .set_algorithm_names([
                'CNN-based',
                'Persistency',
                'Basic-translation',
                'Step-translation',
                'Sequence-translation'
             ]) \
            .set_algorithms(
            [
                ConvolutionalChannelsMovementAlgorithm(),
                PersistencyAlgorithm(),
                BaseTransformation(Transformations.xy_transformation(), trueSkillStatistic.TrueSkillStatistic()),
                MultiImageStepTransformation(Transformations.xy_transformation(),
                                             trueSkillStatistic.TrueSkillStatistic(), 4),
                MultiImageSequenceTransformation(Transformations.xy_transformation(),
                                                 trueSkillStatistic.TrueSkillStatistic(), 4)
            ]
        ) \
            .predict()
        if result is not None:
            uploader.upload_actual(result['files'], '../meteo-out/actual')
            final_result = dict(
                actual=result['files'],
                predicted=r
            )
        else:
            final_result = dict(
                predicted=r
            )
        uploader.upload_results(r, '../output')
        json_result = json.dumps(final_result)
        src = 'last-prediction.json'
        with open(src, 'w') as text_file:
            text_file.write(json_result)
        return json_result
